"""
StockSense AI — Backend
Loads the best model (TimeMachine or iTransformer) that was saved by the
notebook and serves predictions via FastAPI.

Architecture classes are copied verbatim from the notebook so that
`model.load_state_dict(...)` works without any key mismatches.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json as _json
import math
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import uvicorn

# ── paths & constants ──────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "saved_models"
MODEL_INFO_PATH = MODELS_DIR / "best_model_info.json"

WINDOW = 64
FORECAST_HORIZON = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fallback when best_model_info.json is not yet generated
DEFAULT_DIRECTION_SCALE = 0.006528


# =====================================================================
# MODEL ARCHITECTURES  (copied from notebook Cell 4)
# =====================================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features=1, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


def selective_scan(dt, A, x_ssm, B_proj, C_proj, D):
    batch_size, seq_len, d_inner = x_ssm.shape
    d_state = A.shape[1]
    A = A.to(dtype=x_ssm.dtype)
    D = D.to(dtype=x_ssm.dtype)
    h = torch.zeros(batch_size, d_inner, d_state, device=x_ssm.device, dtype=x_ssm.dtype)
    y = torch.zeros_like(x_ssm)
    for t in range(seq_len):
        dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
        dB = (dt[:, t] * x_ssm[:, t]).unsqueeze(-1) * B_proj[:, t].unsqueeze(1)
        h = dA * h + dB
        y[:, t] = (h * C_proj[:, t].unsqueeze(1)).sum(-1) + D * x_ssm[:, t]
    return y

try:
    selective_scan = torch.jit.script(selective_scan)
except Exception:
    pass


class MambaConfig:
    def __init__(self, d_model=32, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            config.d_inner, config.d_inner, config.d_conv,
            groups=config.d_inner, padding=config.d_conv - 1,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def forward(self, x):
        _, seq_len, _ = x.shape
        x_and_res = self.in_proj(x).transpose(1, 2)
        x_ssm, res = torch.split(x_and_res, self.config.d_inner, dim=1)
        x_ssm = F.silu(self.conv1d(x_ssm)[:, :, :seq_len]).transpose(1, 2)
        dt, B_proj, C_proj = torch.split(
            self.x_proj(x_ssm),
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = selective_scan(dt, A, x_ssm, B_proj, C_proj, self.D)
        return self.out_proj(y * F.silu(res.transpose(1, 2)))


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TimeMachine(nn.Module):
    """Mamba/SSM-based TimeMachine — matches notebook exactly."""
    def __init__(self, seq_len=64, pred_len=1, channels=1, d_model=32, num_layers=2):
        super().__init__()
        self.revin = RevIN(num_features=channels)
        self.feature_proj = nn.Linear(channels, d_model)
        self.config = MambaConfig(d_model=d_model)
        self.layers = nn.ModuleList([MambaBlock(self.config) for _ in range(num_layers)])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = self.feature_proj(x)
        for block, norm in zip(self.layers, self.norms):
            x = block(norm(x)) + x
        out = self.fc_out(self.final_norm(x).reshape(x.size(0), -1))
        return out


class iTransformer(nn.Module):
    """Inverted Transformer — matches notebook exactly."""
    def __init__(self, seq_len=64, pred_len=1, channels=1, d_model=64, n_heads=None, e_layers=2):
        super().__init__()
        if n_heads is None:
            n_heads = next(h for h in [8, 4, 2, 1] if d_model % h == 0)
        assert d_model % n_heads == 0
        self.revin = RevIN(num_features=channels)
        self.project = nn.Linear(seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                batch_first=True, dropout=0.1, norm_first=False,
            ),
            num_layers=e_layers,
        )
        self.predict = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        enc = self.encoder(self.project(x.transpose(1, 2)))
        out = self.predict(enc)[:, 0, :]
        return out


# =====================================================================
# FEATURE ENGINEERING (same columns as the notebook)
# =====================================================================

FEATURE_COLUMNS = [
    "Close_Scaled",
    "Return_Scaled",
    "Momentum_3",
    "Momentum_5",
    "Momentum_10",
    "Return_Mean_5",
    "Return_Mean_10",
    "Return_Std_5",
    "Return_Std_10",
    "Price_MA_Diff_5",
    "Price_MA_Diff_10",
    "Price_MA_Diff_20",
    "RSI_14",
]


def normalize_dataset(path, dataset, date_fmt=None):
    if dataset == "7000Stocks":
        df = pd.read_csv(path)
        df = df.rename(columns={"Stock_Name": "Stock"})
    elif dataset == "NIFTY50":
        df = pd.read_csv(path)
        df = df.rename(columns={"Symbol": "Stock"})
    elif dataset == "US500":
        df = pd.read_csv(path)
        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "Name": "Stock",
        })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    df = df[["Date", "Stock", "Open", "High", "Low", "Close", "Volume"]].copy()
    if date_fmt:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format=date_fmt)
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Dataset"] = dataset

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Date", "Stock", "Close"])


def compute_features(stock_df):
    df = stock_df.sort_values("Date").copy()

    scaler = MinMaxScaler()
    df["Close_Scaled"] = scaler.fit_transform(df[["Close"]])

    close = df["Close"].to_numpy()
    returns = np.zeros(len(close))
    returns[1:] = np.diff(np.log(np.clip(close, 1e-8, None)))

    mu = returns.mean()
    sigma = returns.std() if returns.std() > 1e-8 else 1.0
    df["Return_Scaled"] = np.clip((returns - mu) / sigma, -5, 5)

    c = df["Close_Scaled"]
    r = df["Return_Scaled"]

    df["Momentum_3"] = c.diff(3)
    df["Momentum_5"] = c.diff(5)
    df["Momentum_10"] = c.diff(10)
    df["Return_Mean_5"] = r.rolling(5, min_periods=2).mean()
    df["Return_Mean_10"] = r.rolling(10, min_periods=2).mean()
    df["Return_Std_5"] = r.rolling(5, min_periods=2).std()
    df["Return_Std_10"] = r.rolling(10, min_periods=2).std()
    df["Price_MA_Diff_5"] = c - c.rolling(5, min_periods=2).mean()
    df["Price_MA_Diff_10"] = c - c.rolling(10, min_periods=2).mean()
    df["Price_MA_Diff_20"] = c - c.rolling(20, min_periods=2).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=2).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=2).mean()
    rs = gain / (loss + 1e-8)
    df["RSI_14"] = (100 - (100 / (1 + rs))) / 100

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, scaler


# =====================================================================
# DATA LOADING
# =====================================================================

all_data = pd.concat([
    normalize_dataset(DATA_DIR / "7000Stocks_all.csv", "7000Stocks", date_fmt="%Y-%m-%d"),
    normalize_dataset(DATA_DIR / "NIFTY50_all.csv", "NIFTY50", date_fmt="%Y-%m-%d"),
    normalize_dataset(DATA_DIR / "US500_all.csv", "US500", date_fmt="%m-%d-%Y"),
], ignore_index=True)


# =====================================================================
# DYNAMIC MODEL LOADING
# =====================================================================

def load_model_info():
    """Read best_model_info.json exported by the notebook."""
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH, "r") as f:
            info = _json.load(f)
        print(f"[MODEL INFO] Loaded metadata: best_model={info.get('best_model')}, "
              f"direction_scale={info.get('direction_scale')}")
        return info
    return None


def build_model(model_name, info):
    """Instantiate the correct architecture with the right hyperparameters."""
    n_features = len(FEATURE_COLUMNS)
    d_model = info.get("d_model", 32 if model_name == "TimeMachine" else 64) if info else (32 if model_name == "TimeMachine" else 64)
    num_layers = info.get("num_layers", 1) if info else 1
    seq_len = info.get("window", WINDOW) if info else WINDOW

    if model_name == "TimeMachine":
        return TimeMachine(
            seq_len=seq_len, pred_len=1,
            channels=n_features, d_model=d_model,
            num_layers=num_layers,
        )
    elif model_name == "iTransformer":
        return iTransformer(
            seq_len=seq_len, pred_len=1,
            channels=n_features, d_model=d_model,
            e_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_best_model():
    """
    Determine which model to load:
    1. If best_model_info.json exists → use what it says
    2. Else scan saved_models/ for any *_delta_best.pth
    3. Fallback: no model loaded (predictions will error gracefully)
    """
    info = load_model_info()
    direction_scale = DEFAULT_DIRECTION_SCALE

    if info:
        direction_scale = info.get("direction_scale", DEFAULT_DIRECTION_SCALE)
        best_name = info.get("best_model", "TimeMachine")
        model_path = MODELS_DIR / f"{best_name}_delta_best.pth"

        if model_path.exists():
            model = build_model(best_name, info)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE).eval()
            print(f"[MODEL] Loaded {best_name} from {model_path}")
            return model, best_name, direction_scale

    # Fallback: scan for any saved model
    if MODELS_DIR.exists():
        for candidate in ["TimeMachine", "iTransformer"]:
            candidate_path = MODELS_DIR / f"{candidate}_delta_best.pth"
            if candidate_path.exists():
                try:
                    model = build_model(candidate, info)
                    model.load_state_dict(torch.load(candidate_path, map_location=DEVICE))
                    model.to(DEVICE).eval()
                    print(f"[MODEL] Loaded {candidate} (fallback scan) from {candidate_path}")
                    return model, candidate, direction_scale
                except Exception as exc:
                    print(f"[MODEL] Failed to load {candidate}: {exc}")

    print("[MODEL] WARNING: No saved model found. Predictions will not be available.")
    return None, "none", direction_scale


model, active_model_name, direction_scale = load_best_model()


# =====================================================================
# FASTAPI APP
# =====================================================================

app = FastAPI()
app.mount("/static", StaticFiles(directory=APP_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(APP_DIR / "index.html")


@app.get("/api/stocks")
def get_stocks():
    stocks = (
        all_data[["Stock", "Dataset"]]
        .drop_duplicates()
        .sort_values(["Dataset", "Stock"])
        .to_dict("records")
    )
    return stocks


@app.get("/api/predict/{stock}")
def predict_stock(stock: str):
    if model is None:
        return {"error": "No trained model found. Please run the notebook first to train and save a model."}

    # Grab historical local data
    stock_df = all_data[all_data["Stock"] == stock].sort_values("Date")

    if stock_df.empty:
        return {"error": "Stock not found in local dataset."}

    dataset_name = stock_df.iloc[-1]["Dataset"]
    is_live = False
    
    # ── Live yfinance fetching ──
    clean_ticker = stock
    if dataset_name == "7000Stocks" and clean_ticker.lower().endswith(".us"):
        clean_ticker = clean_ticker[:-3]
    elif dataset_name == "NIFTY50":
        clean_ticker = clean_ticker + ".NS"
    
    try:
        # Fetch last 1 year of live data to ensure we have enough history for 64 days + moving averages
        df_yf = yf.download(clean_ticker, period="1y", auto_adjust=True, progress=False)
        
        if not df_yf.empty and len(df_yf) >= WINDOW + 20: 
            df_yf = df_yf.reset_index()
            
            # Flatten multi-index if yfinance returned one
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
                
            df_yf["Stock"] = stock
            df_yf["Dataset"] = dataset_name
            df_yf["Date"] = pd.to_datetime(df_yf["Date"]).dt.tz_localize(None)
            
            # Ensure columns are numeric
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df_yf[col] = pd.to_numeric(df_yf[col], errors="coerce")
                
            df_yf = df_yf.dropna(subset=["Date", "Close"])
            
            if len(df_yf) >= WINDOW:
                # Replace local data with fresh live data!
                stock_df = df_yf[["Date", "Stock", "Open", "High", "Low", "Close", "Volume", "Dataset"]].copy()
                is_live = True
                print(f"[API] Live Data Success: {stock} (ticker: {clean_ticker}) - Last Date: {stock_df.iloc[-1]['Date'].date()}")
            else:
                print(f"[API] Live Data Insufficient after dropna for {clean_ticker}. Falling back to local CSV.")
        else:
            print(f"[API] Live Data Insufficient length for {clean_ticker}. Falling back to local CSV.")
    except Exception as e:
        print(f"[API] Live Data Fetch Failed for {clean_ticker}: {e}. Falling back to local CSV.")

    if len(stock_df) < WINDOW:
        return {"error": "Not enough history for this stock to calculate 64-day window."}

    feature_df, scaler = compute_features(stock_df)
    window = feature_df[FEATURE_COLUMNS].tail(WINDOW).to_numpy(dtype=np.float32)

    x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_delta = float(model(x).cpu().numpy().reshape(-1)[0])

    last = feature_df.iloc[-1]
    last_scaled = float(last["Close_Scaled"])

    pred_scaled = np.clip(last_scaled + pred_delta * direction_scale, 0, 1)
    predicted_close = float(scaler.inverse_transform([[pred_scaled]])[0, 0])

    # ── Build history with AI Model Involvement ──────────────────
    num_hist = min(120, len(feature_df) - WINDOW)
    if num_hist > 0:
        history = feature_df.tail(num_hist).copy()
        
        # Prepare batch for historical predictions
        hist_windows = []
        # We start from the first index that has enough history (WINDOW)
        start_idx = len(feature_df) - num_hist
        for i in range(start_idx, len(feature_df)):
            # For each day 'i', we use the window of features BEFORE it
            win = feature_df[FEATURE_COLUMNS].iloc[i-WINDOW:i].to_numpy(dtype=np.float32)
            hist_windows.append(win)
        
        batch_x = torch.from_numpy(np.stack(hist_windows)).to(DEVICE)
        with torch.no_grad():
            hist_deltas = model(batch_x).cpu().numpy().reshape(-1)
        
        # Each prediction at day 'i' is: ScaledClose(i-1) + ModelDelta
        prev_scaled = feature_df["Close_Scaled"].iloc[start_idx-1 : start_idx-1 + num_hist].to_numpy()
        pred_scaled_hist = np.clip(prev_scaled + hist_deltas * direction_scale, 0, 1)
        
        # Unscale back to real prices
        history["Predicted"] = scaler.inverse_transform(pred_scaled_hist.reshape(-1, 1)).flatten()
    else:
        # Fallback for stocks with very short history
        history = feature_df.tail(120).copy()
        history["Predicted"] = history["Close"].shift(1)

    history["Residual"] = history["Close"] - history["Predicted"]

    # Price movement deltas (matches notebook 7E)
    # We compare today's (actual or predicted) price against YESTERDAY'S ACTUAL close
    history["PriceMovement"] = history["Close"] - history["Close"].shift(1)
    history["PredMovement"] = history["Predicted"] - history["Close"].shift(1)

    history = history.dropna()

    # Direction accuracy data (matches notebook 7F / 7I)
    # actual_dir: 1 if today's Close > yesterday's Close
    actual_dir = (history["Close"] - history["Close"].shift(1) > 0).astype(int)
    # pred_dir: 1 if today's PREDICTION > yesterday's ACTUAL Close
    pred_dir = (history["Predicted"] - history["Close"].shift(1) > 0).astype(int)
    
    dir_match = (actual_dir == pred_dir).astype(int)
    rolling_dir_acc = dir_match.rolling(10, min_periods=1).mean() * 100

    history["ActualDir"] = actual_dir
    history["PredDir"] = pred_dir
    history["DirMatch"] = dir_match
    history["RollingDirAcc"] = rolling_dir_acc

    return {
        "stock": stock,
        "dataset": last["Dataset"],
        "model_name": active_model_name,
        "is_live": is_live,
        "date": str(last["Date"].date()),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "close": float(last["Close"]),
        "volume": int(last["Volume"]) if pd.notna(last["Volume"]) else None,
        "predicted_price": predicted_close,
        "predicted_change": predicted_close - float(last["Close"]),
        "predicted_change_pct": ((predicted_close - float(last["Close"])) / float(last["Close"])) * 100,
        "history": history[["Date", "Close", "Predicted", "Residual",
                            "PriceMovement", "PredMovement",
                            "ActualDir", "PredDir", "DirMatch", "RollingDirAcc"]]
            .assign(Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d"))
            .to_dict("records")
    }


if __name__ == "__main__":
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
