
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tkinter import *
import tkinter
from tkinter import filedialog, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from sklearn.preprocessing import label_binarize

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_score, accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import joblib
from tao_tree import TAOTreeClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, LSTM, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from transformers import Wav2Vec2FeatureExtractor, HubertModel
from huggingface_hub import snapshot_download

import torch


path = None
model_folder = "model"
os.makedirs(model_folder, exist_ok=True)
categories = []

# Files used by your code
X_file = os.path.join(model_folder, "X.npy")
Y_file = os.path.join(model_folder, "Y.npy")
X_hubert_file = os.path.join(model_folder, "X_hubert.npy")
Y_hubert_file = os.path.join(model_folder, "Y_hubert.npy")
Tao_hubert_path = os.path.join(model_folder, "Tao_on_HuBERT.pkl")
logreg_path = os.path.join(model_folder, "LogisticRegression.pkl")
lda_path = os.path.join(model_folder, "LDAClassifier.pkl")

# Labels used in original notebook
LABELS = ['Air leak', 'Background noise', 'idling', 'normal', 'Oil leak']

# variables to hold data/models
X = None
Y = None
X_train = X_test = y_train = y_test = None
X_hubert = None
Y_hubert = None
Tao_hubert = None

HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = None
hubert = None

# --------------------------
# UI helpers
# --------------------------
def ui_clear_and_log(s):
    text.delete('1.0', END)
    text.insert(END, s + "\n")
    text.update()

def ui_append(s):
    text.insert(END, s + "\n")
    text.see(END)

# --------------------------
# Button implementations
# --------------------------

def upload_audio_dataset():
    """Select the dataset folder (root with labeled subfolders)."""
    global path, categories
    folder = filedialog.askdirectory(initialdir=".")
    if not folder:
        return
    path = folder
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    ui_clear_and_log(f"Dataset folder selected: {path}\nCategories found: {categories}")

def mfcc_feature_extraction():
    global X, Y, categories
    try:
        if os.path.exists(X_file) and os.path.exists(Y_file):
            X = np.load(X_file, allow_pickle=True)
            Y = np.load(Y_file, allow_pickle=True)
            ui_clear_and_log("X and Y arrays loaded successfully.")
            sns.countplot(x=Y)
            plt.show()
            return
    except Exception as e:
        ui_append(f"Load pre-saved X/Y failed: {e}")

    # If not found, run original extraction code (kept same)
    ui_append("Starting feature extraction (...")
    X_list = []
    Y_list = []
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for root, dirs, files in os.walk(path):
        for fname in files:
            if 'Thumbs.db' in fname:
                continue
            if not fname.lower().endswith(".wav"):
                continue
            audio_path = os.path.join(root, fname)
            try:
                audio, sr = librosa.load(audio_path, sr=None)

                # EXACT features from your notebook
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
                rms = librosa.feature.rms(y=audio)
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
                tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

                features = np.concatenate((
                    mfccs.flatten(), 
                    zero_crossings.flatten(),
                    spectral_centroid.flatten(),
                    chroma.flatten(),
                    spectral_bandwidth.flatten(),
                    spectral_contrast.flatten(),
                    rms.flatten(),
                    rolloff.flatten(),
                    tonnetz.flatten(),
                    mel_spectrogram.flatten()
                ), axis=0)

                max_length = 3000
                if len(features) < max_length:
                    features = np.pad(features, (0, max_length - len(features)), mode='constant')
                else:
                    features = features[:max_length]

                X_list.append(features)
                label_name = os.path.basename(root)
                Y_list.append(categories.index(label_name))

                ui_append(f"Extracted features: {audio_path}")
            except Exception as e:
                ui_append(f"Failed to extract {audio_path}: {e}")

    if len(X_list) == 0:
        ui_append("No features extracted.")
        return

    X = np.array(X_list)
    Y = np.array(Y_list)
    np.save(X_file, X)
    np.save(Y_file, Y)
    ui_append(f"Saved X and Y to {X_file} and {Y_file}")
    sns.countplot(x=Y)
    plt.show()

def data_splitting():
    global X, Y, X_train, X_test, y_train, y_test, num_classes, y_train_cat, y_test_cat
    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file, allow_pickle=True)
        Y = np.load(Y_file, allow_pickle=True)
        ui_append("Loaded saved X & Y arrays.")
    else:
        ui_append(" Run MFCC feature extraction.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    ui_clear_and_log(f"Data split done. X_train: {X_train.shape}, X_test: {X_test.shape}")
    sns.countplot(x=y_train)
    plt.title("Train distribution")
    plt.show()

def train_logistic():
    
    global X_train, X_test, y_train, y_test
    if 'X_train' not in globals():
        ui_append("Data not split. Run Data splitting first.")
        return

    if os.path.exists(logreg_path):
        logistic_model = joblib.load(logreg_path)
        ui_append("Loaded existing LogisticRegression model.")
    else:
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        joblib.dump(logistic_model, logreg_path)
        ui_append("Trained and saved LogisticRegression model.")

    y_pred_logreg = logistic_model.predict(X_test)

    # compute metrics
    acc = accuracy_score(y_test, y_pred_logreg) * 100
    prec = precision_score(y_test, y_pred_logreg, average='macro', zero_division=0) * 100
    rec = recall_score(y_test, y_pred_logreg, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred_logreg, average='macro', zero_division=0) * 100

    ui_append(f"Logistic — Accuracy: {acc:.2f}%  Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
    ui_append("Logistic Regression detailed report:")
    ui_append(classification_report(y_test, y_pred_logreg, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_logreg)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ROC curve 
    try:
        if hasattr(logistic_model, "predict_proba"):
            y_score = logistic_model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=np.arange(y_score.shape[1]))
            plt.figure(figsize=(8,6))
            for i in range(y_score.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={roc_auc:.2f})")
            plt.plot([0,1],[0,1],'--')
            plt.title("Logistic Regression ROC (per-class)")
            plt.legend()
            plt.show()
    except Exception as e:
        ui_append("ROC plotting failed for Logistic: " + str(e))

def train_lda():
    """
    """
    global X_train, X_test, y_train, y_test
    if 'X_train' not in globals():
        ui_append("Data not split. Run Data splitting first.")
        return

    if os.path.exists(lda_path):
        lda_model = joblib.load(lda_path)
        ui_append("Loaded existing LDA model.")
    else:
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_train, y_train)
        joblib.dump(lda_model, lda_path)
        ui_append("Trained and saved LDA model.")

    y_pred_lda = lda_model.predict(X_test)

    # compute metrics
    acc = accuracy_score(y_test, y_pred_lda) * 100
    prec = precision_score(y_test, y_pred_lda, average='macro', zero_division=0) * 100
    rec = recall_score(y_test, y_pred_lda, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred_lda, average='macro', zero_division=0) * 100

    ui_append(f"LDA — Accuracy: {acc:.2f}%  Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
    ui_append("LDA detailed report:")
    ui_append(classification_report(y_test, y_pred_lda, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_lda)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("LDA Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ROC if possible
    try:
        if hasattr(lda_model, "predict_proba"):
            y_score = lda_model.predict_proba(X_test)
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.arange(y_score.shape[1]))
            plt.figure(figsize=(8,6))
            for i in range(y_score.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={roc_auc:.2f})")
            plt.plot([0,1],[0,1],'--')
            plt.title("LDA ROC (per-class)")
            plt.legend()
            plt.show()
        else:
            # fallback: try transform -> softmax-ish scores for ROC plotting (not exact probabilities)
            try:
                trans = lda_model.transform(X_test)
                from scipy.special import softmax
                if trans.ndim == 1:
                    trans = trans.reshape(-1,1)
                n_classes = len(np.unique(y_test))
                if trans.shape[1] < n_classes:
                    pad = np.zeros((trans.shape[0], n_classes - trans.shape[1]))
                    trans_full = np.hstack([trans, pad])
                else:
                    trans_full = trans[:, :n_classes]
                y_score = softmax(trans_full, axis=1)
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=np.arange(y_score.shape[1]))
                plt.figure(figsize=(8,6))
                for i in range(y_score.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={roc_auc:.2f})")
                plt.plot([0,1],[0,1],'--')
                plt.title("LDA ROC (approx, from transform)")
                plt.legend()
                plt.show()
            except Exception:
                ui_append("LDA ROC plotting skipped (no probabilities available).")
    except Exception as e:
        ui_append("ROC plotting failed for LDA: " + str(e))

def init_hubert_and_embed(wav, sr, target_sr=16000):
    """Helper replicating your hubert_embed function block."""
    global feature_extractor, hubert, device
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    if feature_extractor is None or hubert is None:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
        hubert = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(device)
        hubert.eval()
    inputs = feature_extractor(wav, sampling_rate=target_sr, return_tensors="pt")
    inputs = inputs["input_values"].to(device)
    with torch.no_grad():
        out = hubert(inputs).last_hidden_state
        emb = out.mean(dim=1).squeeze().cpu().numpy()
    return emb

def train_hubert_Tao():
    """
    """
    global X_hubert, Y_hubert, Tao_hubert
    # Check saved hubert features first
    if os.path.exists(X_hubert_file) and os.path.exists(Y_hubert_file):
        X_hubert = np.load(X_hubert_file)
        Y_hubert = np.load(Y_hubert_file)
        ui_append("Loaded saved HuBERT features.")
    else:
        ui_append("Extracting HuBERT features (original notebook logic). This may be slow on CPU.")
        # load models and extract
        feature_extractor_local = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
        hubert_local = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(device)
        hubert_local.eval()
        audio_embeddings = []
        audio_labels = []
        for root, dirs, files in os.walk(path):
            for fname in files:
                if not fname.lower().endswith(".wav"): continue
                if fname == "Thumbs.db": continue
                fpath = os.path.join(root, fname)
                wav, sr = librosa.load(fpath, sr=None)
                # embed
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                inputs = feature_extractor_local(wav, sampling_rate=sr, return_tensors="pt")["input_values"].to(device)
                with torch.no_grad():
                    out = hubert_local(inputs).last_hidden_state
                    emb = out.mean(dim=1).squeeze().cpu().numpy()
                audio_embeddings.append(emb)
                label_name = os.path.basename(root)
                audio_labels.append(categories.index(label_name))
                ui_append("Extracted: " + fname)
        if len(audio_embeddings) == 0:
            ui_append("No HuBERT embeddings created.")
            return
        X_hubert = np.vstack(audio_embeddings)
        Y_hubert = np.array(audio_labels)
        np.save(X_hubert_file, X_hubert)
        np.save(Y_hubert_file, Y_hubert)
        ui_append("Saved HuBERT feature files!")

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_hubert, Y_hubert,
        test_size=0.2,
        random_state=42,
        stratify=Y_hubert
    )

    # CHECK IF MODEL EXISTS
    if os.path.exists(Tao_hubert_path):
        ui_append("Loading saved Tao model...")
        Tao_hubert = joblib.load(Tao_hubert_path)
    else:
        ui_append("Training Tao on HuBERT features...")
        Tao_hubert = TAOTreeClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
)

        Tao_hubert.fit(X_train_h, y_train_h)
        joblib.dump(Tao_hubert, Tao_hubert_path)
        ui_append("Saved Tao model to: " + Tao_hubert_path)

    ui_append("Evaluating Tao on HuBERT features...")
    y_pred_h = Tao_hubert.predict(X_test_h)

    # compute metrics
    acc = accuracy_score(y_test_h, y_pred_h) * 100
    prec = precision_score(y_test_h, y_pred_h, average='macro', zero_division=0) * 100
    rec = recall_score(y_test_h, y_pred_h, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test_h, y_pred_h, average='macro', zero_division=0) * 100

    ui_append(f"HuBERT+Tao — Accuracy: {acc:.2f}%  Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")
    ui_append(classification_report(y_test_h, y_pred_h, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_test_h, y_pred_h)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("HuBERT + Tao Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ROC for Tao_hubert if predict_proba available
    try:
        if hasattr(Tao_hubert, "predict_proba"):
            y_score = Tao_hubert.predict_proba(X_test_h)
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test_h, classes=np.arange(y_score.shape[1]))
            plt.figure(figsize=(8,6))
            for i in range(y_score.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={roc_auc:.2f})")
            plt.plot([0,1],[0,1],'--')
            plt.title("HuBERT+RF ROC (per-class)")
            plt.legend()
            plt.show()
    except Exception as e:
        ui_append("ROC plotting failed for HuBERT+Tao: " + str(e))

def predict_audio():
    """Predict a single audio file using available saved models. Shows waveform and predicted label (original logic)."""
    filetypes = [("WAV files", "*.wav"), ("All files", "*.*")]
    filename = filedialog.askopenfilename(initialdir=".", filetypes=filetypes)
    if not filename:
        return

    # Prefer Tao_on_HuBERT if available
    if os.path.exists(Tao_hubert_path):
        Tao_hubert_local = joblib.load(Tao_hubert_path)
        # compute embedding using hubert model if needed
        try:
            # ensure hubert extraction code (same as notebook)
            feature_extractor_local = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
            hubert_local = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(device)
            hubert_local.eval()
            y, sr = librosa.load(filename, sr=None)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            inputs = feature_extractor_local(y, sampling_rate=sr, return_tensors="pt")["input_values"].to(device)
            with torch.no_grad():
                out = hubert_local(inputs).last_hidden_state
                emb = out.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)
            pred = Tao_hubert_local.predict(emb)[0]
            label = LABELS[pred] if pred < len(LABELS) else str(pred)
            ui_clear_and_log(f"{os.path.basename(filename)}  →  {label}")
            plt.figure(figsize=(10,3))
            yvis, srvis = librosa.load(filename, sr=None)
            librosa.display.waveshow(yvis, sr=srvis)
            plt.title(f"Predicted: {label}")
            plt.tight_layout()
            plt.show()
            return
        except Exception as e:
            ui_append("RF_on_HuBERT prediction failed: " + str(e))

    

# --------------------------
root = Tk()
root.title("Deep acoustic monitoring of cutting tool wear in machining process")
root.geometry("1000x650")
root.configure(bg='#f0f8ff')

# Robust background image loader (replace your existing BG_IMAGE_PATH block)
BG_IMAGE_PATH = r"background.jpg"

if not os.path.exists(BG_IMAGE_PATH):
    print(f"[BG] Background image not found at: {BG_IMAGE_PATH}")
else:
    try:
        _bg_orig = Image.open(BG_IMAGE_PATH).convert("RGBA")
    except Exception as e:
        print(f"[BG] Failed to open background image: {e}")
    else:
        # create a Label that will hold the image (covering the whole window)
        bg_label = Label(root)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # keep original and current photo on root so GC doesn't collect them
        root._bg_orig = _bg_orig
        root._bg_photo = None

        def _resize_bg(event=None):
            # sometimes configure fires with tiny sizes; guard against that
            w = max(1, root.winfo_width())
            h = max(1, root.winfo_height())
            try:
                # use high-quality resampling
                _resized = root._bg_orig.resize((w, h), Image.LANCZOS)
                root._bg_photo = ImageTk.PhotoImage(_resized)
                bg_label.configure(image=root._bg_photo)
                bg_label.lower()   # send background behind other widgets
            except Exception as e:
                # Print error to console and UI textbox for debugging
                print(f"[BG] Error resizing background image: {e}")
                try:
                    ui_append(f"Background resize error: {e}")
                except:
                    pass

        # initialize image to current geometry after Tk has finished geometry setup
        root.update_idletasks()
        _resize_bg()

        # update image whenever window size changes
        root.bind("<Configure>", _resize_bg)
        print("[BG] Background image loaded and will resize with window.")


Label(root, text="Deep acoustic monitoring of cutting tool wear in machining process", bg='#003366', fg='white', font=('times',16,'bold')).pack(fill='x', pady=6)

btn_font = ('times',12,'bold')
Button(root, text="Upload audio dataset", width=22, command=upload_audio_dataset, font=btn_font).place(x=40, y=70)
Button(root, text="MFCC feature extraction", width=22, command=mfcc_feature_extraction, font=btn_font).place(x=300, y=70)
Button(root, text="Data splitting", width=22, command=data_splitting, font=btn_font).place(x=560, y=70)

Button(root, text="Train Logistic", width=22, command=train_logistic, font=btn_font).place(x=40, y=140)
Button(root, text="Train LDA", width=22, command=train_lda, font=btn_font).place(x=300, y=140)
Button(root, text="Train Hybrid hubert with Tao", width=30, command=train_hubert_Tao, font=btn_font).place(x=560, y=140)

Button(root, text="Predict audio", width=22, command=predict_audio, font=btn_font).place(x=360, y=200)
Button(root, text="Exit", width=10, command=root.destroy, font=btn_font).place(x=760, y=200)

# Output text box with Times New Roman, bold, size 12
output_font = ("Times New Roman", 12, "bold")
text = Text(root, height=15, width=80, font=output_font)
text.place(x=350, y=400)

root.mainloop()
