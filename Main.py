from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from tkinter import messagebox, simpledialog, filedialog, Tk, END, Text

# =============================
# 📦 Core Libraries
# =============================
import os
import cv2
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 🎵 Audio Processing
# =============================
import librosa
import librosa.display

# =============================
# ⚙️ Data Preprocessing & Utilities
# =============================
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# =============================
# 📈 Evaluation Metrics
# =============================
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)

# =============================
# 🧠 Machine Learning Models
# =============================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)

# =============================
# 🤖 Deep Learning (Keras/TensorFlow)
# =============================
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, GRU, Bidirectional,
    Flatten, Dense, Dropout, Input, concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

  
accuracy = []
precision = []
recall = []
fscore = []

model_folder = "Model"
os.makedirs(model_folder, exist_ok=True)

results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

def Upload_Dataset():
    global filename, categories, num_classes
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, 'Dataset loaded successfully\n')
    text.insert(END, f"Selected dataset path: {filename}\n")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    print(file_path)
    features = []

    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # 3. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features.extend(np.mean(mel, axis=1))

    # 4. Spectral Centroid
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # 5. Spectral Bandwidth
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # 6. Spectral Rolloff
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # 7. Zero Crossing Rate
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    # 8. RMS Energy
    features.append(np.mean(librosa.feature.rms(y=y)))

    # 9. Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features.extend(np.mean(tonnetz, axis=1))

    # 10. Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spec_contrast, axis=1))

    return np.array(features)

def load_dataset(X_file, Y1_file, Y2_file, model_folder, dataset_path):
    mix_folder = os.path.join(dataset_path, "Mix3")
    csv_path = os.path.join(dataset_path, "Mix.csv")

    if not os.path.exists(mix_folder):
        raise FileNotFoundError(f"'Mix3' folder not found in {dataset_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'Mix.csv' not found in {dataset_path}")

    df = pd.read_csv(csv_path)

    X = []
    Y1 = []  # Heart Sound Type
    Y2 = []  # Lung Sound Type

    for _, row in df.iterrows():
        mixed_id = str(row['Mixed Sound ID'])
        wav_file = os.path.join(mix_folder, f"{mixed_id}.wav")

        if os.path.exists(wav_file):
            features = extract_features(wav_file)
            X.append(features)
            Y1.append(row['Heart Sound Type'])
            Y2.append(row['Lung Sound Type'])
        else:
            print(f"Warning: File not found for ID {mixed_id}")

    # Save as .npy
    np.save(X_file, np.array(X))
    np.save(Y1_file, np.array(Y1))
    np.save(Y2_file, np.array(Y2))

    print(f"Feature extraction complete. Saved: {X_file}, {Y1_file}, {Y2_file}")
    return np.array(X), np.array(Y1), np.array(Y2)

def EDA():
    global text, results_folder

    # Ensure results folder exists
    if not os.path.exists(results_folder):
        text.delete('1.0', END)
        text.insert(END, "❌ Results folder not found.\n")
        return

    # Get all files in results folder that start with "EDA_"
    eda_files = sorted([
        os.path.join(results_folder, f)
        for f in os.listdir(results_folder)
        if f.startswith("EDA_") and f.lower().endswith((".png", ".jpg"))
    ])

    text.delete('1.0', END)

    if not eda_files:
        text.insert(END, "⚠️ EDA is not done, Plots are generating and takes time.\n")
        EDA_Saved()
        return

    text.insert(END, f"✅ Found {len(eda_files)} EDA result file(s):\n\n")
    for f in eda_files:
        text.insert(END, f"Displaying: {os.path.basename(f)}\n")
        img = cv2.imread(f)
        if img is None:
            text.insert(END, f"⚠️ Could not read {f}\n")
            continue

        # Resize for better viewing if too large
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            img = cv2.resize(img, (1200, int(height * scale)))

        # Display the image
        cv2.imshow(f"EDA Visualization - {os.path.basename(f)}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    text.insert(END, "\n✅ All available EDA visualizations displayed.\n")
    
def EDA_Saved():
    global filename, text, results_folder

    mix_folder = os.path.join(filename, "Mix3")
    csv_path = os.path.join(filename, "Mix.csv")

    if not os.path.exists(mix_folder):
        text.insert(END, f" 'Mix3' folder not found in: {filename}\n")
        return
    if not os.path.exists(csv_path):
        text.insert(END, f" 'Mix.csv' file not found in: {filename}\n")
        return

    df = pd.read_csv(csv_path)
    text.insert(END, f"✅ Metadata loaded from: {csv_path}\n")
    text.insert(END, f"Total Records: {len(df)}\n\n")

    # Group by Heart Sound Type (you can change this to 'Lung Sound Type' if needed)
    unique_classes = df['Heart Sound Type'].unique()
    text.insert(END, f"Unique Heart Sound Classes Found: {list(unique_classes)}\n\n")

    for cls in unique_classes:
        subset = df[df['Heart Sound Type'] == cls]
        if subset.empty:
            continue
        
        # Pick one representative sample for this class
        sample_row = subset.sample(n=1, random_state=42).iloc[0]
        mixed_id = str(sample_row['Mixed Sound ID'])
        wav_file = os.path.join(mix_folder, f"{mixed_id}.wav")

        if not os.path.exists(wav_file):
            text.insert(END, f"⚠️ Missing file: {mixed_id}.wav for class {cls}\n")
            continue

        try:
            y, sr = librosa.load(wav_file, sr=16000)  # Resample to 16kHz
            y_harm = librosa.effects.harmonic(y)

            fig, axes = plt.subplots(5, 2, figsize=(14, 16))
            fig.subplots_adjust(hspace=0.5, wspace=0.3)

            # --- 1. MFCC ---
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            img = librosa.display.specshow(mfcc, x_axis='time', ax=axes[0, 0])
            axes[0, 0].set_title(f"{cls} - MFCC")
            fig.colorbar(img, ax=axes[0, 0], format='%+2.0f dB')

            # --- 2. Chroma ---
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=axes[0, 1])
                axes[0, 1].set_title("Chroma")
                fig.colorbar(img, ax=axes[0, 1])
            except librosa.util.exceptions.ParameterError:
                axes[0, 1].set_title("Chroma (Skipped - Low SR)")

            # --- 3. Mel Spectrogram ---
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), x_axis='time', y_axis='mel', ax=axes[1, 0])
            axes[1, 0].set_title("Mel Spectrogram")
            fig.colorbar(img, ax=axes[1, 0], format='%+2.0f dB')

            # --- 4. Spectral Centroid ---
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            axes[1, 1].plot(spec_centroid, color='r')
            axes[1, 1].set_title("Spectral Centroid")
            axes[1, 1].grid(True)

            # --- 5. Spectral Bandwidth ---
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            axes[2, 0].plot(spec_bandwidth, color='g')
            axes[2, 0].set_title("Spectral Bandwidth")
            axes[2, 0].grid(True)

            # --- 6. Spectral Rolloff ---
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            axes[2, 1].plot(spec_rolloff, color='c')
            axes[2, 1].set_title("Spectral Rolloff")
            axes[2, 1].grid(True)

            # --- 7. Zero Crossing Rate ---
            zcr = librosa.feature.zero_crossing_rate(y)
            axes[3, 0].plot(zcr[0], color='m')
            axes[3, 0].set_title("Zero Crossing Rate")
            axes[3, 0].grid(True)

            # --- 8. RMS Energy ---
            rms = librosa.feature.rms(y=y)[0]
            axes[3, 1].plot(rms, color='b')
            axes[3, 1].set_title("Root Mean Square Energy (RMS)")
            axes[3, 1].grid(True)

            # --- 9. Tonnetz ---
            try:
                tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
                img = librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time', ax=axes[4, 0])
                axes[4, 0].set_title("Tonnetz (Tonal Centroid Features)")
                fig.colorbar(img, ax=axes[4, 0])
            except librosa.util.exceptions.ParameterError:
                axes[4, 0].set_title("Tonnetz (Skipped - Low SR)")

            # --- 10. Spectral Contrast ---
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            img = librosa.display.specshow(spec_contrast, x_axis='time', ax=axes[4, 1])
            axes[4, 1].set_title("Spectral Contrast")
            fig.colorbar(img, ax=axes[4, 1])

            # Save figure
            fig_filename = f"EDA_HeartSound_{cls.replace(' ', '_')}.png"
            fig_path = os.path.join(results_folder, fig_filename)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show(fig)

            text.insert(END, f"✅ Saved EDA visualization for class: {cls}\n")

        except Exception as e:
            text.insert(END, f"⚠️ Error processing {mixed_id} ({cls}): {str(e)}\n")
            continue

    text.insert(END, "\n✅ All class-wise EDA visualizations saved in the 'results' folder.\n")

def Preprocess_Dataset():
    global X, Y1, Y2, filename, model_folder
    filepath = Path(filename)
    last_folder = filepath.name

    # File paths for preprocessed data
    X_file = os.path.join(model_folder, "X1n.txt.npy")
    Y1_file = os.path.join(model_folder, "Y1n.txt.npy")
    Y2_file = os.path.join(model_folder, "Y2n.txt.npy")

    # Check if already preprocessed files exist
    if os.path.exists(X_file) and os.path.exists(Y1_file) and os.path.exists(Y2_file):
        X = np.load(X_file, allow_pickle=True)
        Y1 = np.load(Y1_file, allow_pickle=True)
        Y2 = np.load(Y2_file, allow_pickle=True)
        text.insert(END, "Loaded existing preprocessed dataset files.\n")
    else:
        # Process and extract features + targets
        X, Y1, Y2 = load_dataset(X_file, Y1_file, Y2_file, model_folder, filename)
        text.insert(END, "Feature extraction and dataset creation completed.\n")

    # Convert lists to numpy arrays if needed
    X = np.array(X)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)


    
    # Display dataset info
    text.insert(END, f"Preprocessing and Feature Extraction completed for Dataset: {filename}\n\n")
    text.insert(END, f"Input Feature Set Shape: {X.shape}\n")
    text.insert(END, f"Heart Sound Labels Count: {len(np.unique(Y1))}\n")
    text.insert(END, f"Lung Sound Labels Count: {len(np.unique(Y2))}\n\n")

    print("Preprocessing completed successfully.")
    
def Train_Test_Splitting():
    global X, Y1, Y2
    global x_train, y1_train, y2_train, x_test, y1_test, y2_test
    global categories_heart, categories_lung  # class-name lists for metrics
    global model_folder, text

    # --- consistent shuffle ---
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)

    X  = X[indices]
    Y1 = Y1[indices]
    Y2 = Y2[indices]

    # --- single split using indices to keep both targets aligned ---
    all_idx = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, shuffle=True)

    x_train, x_test = X[train_idx], X[test_idx]
    y1_train, y1_test = Y1[train_idx], Y1[test_idx]
    y2_train, y2_test = Y2[train_idx], Y2[test_idx]

    # --- class-name variables for metrics ---
    categories_heart = np.unique(Y1).astype(str).tolist()
    categories_lung  = np.unique(Y2).astype(str).tolist()

    # --- info ---
    text.delete('1.0', END)
    text.insert(END, f"Total Samples: {X.shape[0]}\n\n")
    text.insert(END, f"Training Samples: {x_train.shape[0]}\n")
    text.insert(END, f"Testing Samples: {x_test.shape[0]}\n\n")
    text.insert(END, f"Heart Sound Classes: {categories_heart}\n")
    text.insert(END, f"Lung Sound Classes: {categories_lung}\n\n") 
    
def Calculate_Metrics(categories, algorithm, predict, y_test,y_score):

    if not os.path.exists('results'):
        os.makedirs('results')
        
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure() 
    ax = sns.heatmap(
        conf_matrix,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,
        cmap="viridis",
        fmt="g"
    )

    ax.set_ylim([0, len(categories)])  # Keeps all rows visible
    plt.title(algorithm + " Confusion Matrix", fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    plt.tight_layout()  # Adjusts subplots to fit within figure area

    # ✅ Save figure without cropping anything
    plt.savefig(
        f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches='tight'  # Ensures full content is saved
    )
    plt.show()

    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=range(len(categories)))
        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(len(categories)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            

        plt.figure(figsize=(10, 8))
        for i in range(len(categories)):
            plt.plot(fpr[i], tpr[i], label=f'Class {categories[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.title(f"{algorithm} ROC Curves (One-vs-Rest)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.show()
        


def existing_classifier1():
    """Random Forest"""
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test
    global categories_heart, categories_lung, text

    text.delete('1.0', END)

    # Heart
    rf_h = QuadraticDiscriminantAnalysis(reg_param=0.9)
    rf_h.fit(x_train, y1_train)
    y1_pred  = rf_h.predict(x_test)
    y1_score = rf_h.predict_proba(x_test)
    y1_score = None
    Calculate_Metrics(categories_heart, "QDA Heart", y1_pred, y1_test, y1_score)

    # Lung
    rf_l = QuadraticDiscriminantAnalysis(reg_param=0.3)
    rf_l.fit(x_train, y2_train)
    y2_pred  = rf_l.predict(x_test)
    y2_score = rf_l.predict_proba(x_test)
    y2_score = None
    Calculate_Metrics(categories_lung, "QDA Lung", y2_pred, y2_test, y2_score)


def existing_classifier2():
    """Gradient Boosting"""
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test
    global categories_heart, categories_lung, text

    text.delete('1.0', END)

    gb_h = GradientBoostingClassifier(n_estimators=15, learning_rate=0.05, random_state=42)
    gb_h.fit(x_train, y1_train)
    y1_pred  = gb_h.predict(x_test)
    y1_score = gb_h.predict_proba(x_test)
    y1_score = None
    Calculate_Metrics(categories_heart, "Gradient Boosting Heart", y1_pred, y1_test, y1_score)

    gb_l = GradientBoostingClassifier(n_estimators=15, learning_rate=0.05, random_state=42)
    gb_l.fit(x_train, y2_train)
    y2_pred  = gb_l.predict(x_test)
    y2_score = gb_l.predict_proba(x_test)
    y2_score = None
    Calculate_Metrics(categories_lung, "Gradient Boosting Lung", y2_pred, y2_test, y2_score)


def existing_classifier3():
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test
    global categories_heart, categories_lung, text

    text.delete('1.0', END)
    nb_h = GaussianNB()
    nb_h.fit(x_train, y1_train)
    y1_pred_nb  = nb_h.predict(x_test)
    y1_score_nb = nb_h.predict_proba(x_test)
    y1_score = None
    Calculate_Metrics(categories_heart, "Naive Bayes Heart", y1_pred_nb, y1_test, y1_score_nb)
    
    nb_l = GaussianNB()
    nb_l.fit(x_train, y2_train)
    y2_pred_nb  = nb_l.predict(x_test)
    y2_score_nb = nb_l.predict_proba(x_test)
    y2_score = None   
    Calculate_Metrics(categories_lung, "Naive Bayes Lung", y2_pred_nb, y2_test, y2_score_nb)


def existing_classifier4():
    """Logistic Regression"""
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test
    global categories_heart, categories_lung, text

    text.delete('1.0', END)

    # Heart
    log_h = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='auto')
    log_h.fit(x_train, y1_train)
    y1_pred  = log_h.predict(x_test)
    y1_score = log_h.predict_proba(x_test)
    y1_score = None
    Calculate_Metrics(categories_heart, "Logistic Regression Heart", y1_pred, y1_test, y1_score)


    # Lung
    log_l = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='auto')
    log_l.fit(x_train, y2_train)
    y2_pred  = log_l.predict(x_test)
    y2_score = log_l.predict_proba(x_test)
    y2_score = None
    Calculate_Metrics(categories_lung, "Logistic Regression Lung", y2_pred, y2_test, y2_score)





def bicnn_bigru_classifier():
    global x_train, x_test, y1_train, y1_test, y2_train, y2_test
    global categories_heart, categories_lung, text

    text.delete('1.0', END)
    text.insert(END, "Starting BiCNN-BiGRU Training for Heart and Lung Sound Types...\n\n")

    # Utility function to train and evaluate a model for one target variable
    def train_model(X_train, X_test, Y_train, Y_test, categories, label_name):
        # Encode labels
        le = LabelEncoder()
        Y_train_encoded = le.fit_transform(Y_train)
        Y_test_encoded = le.transform(Y_test)
        num_classes = len(np.unique(Y_train_encoded))

        os.makedirs("model", exist_ok=True)
        encoder_path = f"model/{label_name}_label_encoder.pkl"
        model_path = f"model/{label_name}_BiCNN_BiGRU.h5"
        history_path = f"model/{label_name}_BiCNN_BiGRU_history.pkl"

        # Save label encoder
        with open(encoder_path, "wb") as f:
            pickle.dump(le, f)

        # Reshape input for CNN-GRU
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        Y_train_cat = to_categorical(Y_train_encoded, num_classes=num_classes)
        Y_test_cat = to_categorical(Y_test_encoded, num_classes=num_classes)

        # Load or build model
        if os.path.exists(model_path):
            model = load_model(model_path)
            with open(history_path, "rb") as f:
                history = pickle.load(f)
            text.insert(END, f"Loaded pre-trained BiCNN-BiGRU model for {label_name}.\n\n")
        else:
            # --- BiCNN-BiGRU architecture ---
            inp = Input(shape=(X_train.shape[1], 1))

            # Parallel CNN branches (BiCNN)
            conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
            conv1 = MaxPooling1D(pool_size=2)(conv1)
            conv2 = Conv1D(128, kernel_size=5, activation='relu', padding='same')(inp)
            conv2 = MaxPooling1D(pool_size=2)(conv2)

            merged = concatenate([conv1, conv2])

            # Bidirectional GRU layers
            bigru = Bidirectional(GRU(128, return_sequences=False, dropout=0.3))(merged)

            # Dense layers
            dense = Dense(256, activation='relu')(bigru)
            dense = Dropout(0.4)(dense)
            output = Dense(num_classes, activation='softmax')(dense)

            model = Model(inputs=inp, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Train
            history = model.fit(
                X_train_reshaped, Y_train_cat,
                validation_split=0.2, epochs=80, batch_size=32, verbose=2
            )

            # Save model and history
            model.save(model_path)
            with open(history_path, "wb") as f:
                pickle.dump(history.history, f)

            history = history.history
            text.insert(END, f"✅ Trained and saved BiCNN-BiGRU model for {label_name}.\n\n")

        # --- Evaluate ---
        y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
        y_score = model.predict(X_test_reshaped)

        # --- Plot accuracy and loss ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{label_name} BiCNN-BiGRU Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{label_name} BiCNN-BiGRU Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # --- Metrics ---
        Calculate_Metrics(categories, f"BiCNN-BiGRU {label_name}", y_pred, Y_test_encoded, y_score)

    # Train model separately for Heart and Lung sounds
    text.insert(END, "Training BiCNN-BiGRU Model for Heart Sound Type...\n")
    train_model(x_train, x_test, y1_train, y1_test, categories_heart, "Heart_Sound")

    text.insert(END, "\nTraining BiCNN-BiGRU Model for Lung Sound Type...\n")
    train_model(x_train, x_test, y2_train, y2_test, categories_lung, "Lung_Sound")

    text.insert(END, "\n✅ BiCNN-BiGRU training and evaluation completed for both Heart and Lung sound types.\n")


def Prediction():
    global text

    # --- Select audio file ---
    filename = filedialog.askopenfilename(initialdir="Test", filetypes=[("Audio Files", "*.wav *.mp3")])
    text.delete('1.0', END)
    if not filename:
        text.insert(END, "No file selected.\n")
        return

    text.insert(END, f"{filename} Loaded\n")

    # --- Extract audio features ---
    features = extract_features(filename)
    features = features.reshape(1, -1)
    features_reshaped = features.reshape((1, features.shape[1], 1))

    # --- Define models and encoders ---
    model_info = {
        "Heart": {
            "model": "model/Heart_Sound_BiCNN_BiGRU.h5",
            "encoder": "model/Heart_Sound_label_encoder.pkl"
        },
        "Lung": {
            "model": "model/Lung_Sound_BiCNN_BiGRU.h5",
            "encoder": "model/Lung_Sound_label_encoder.pkl"
        }
    }

    predictions = {}

    # --- Predict for both Heart and Lung ---
    for label_name, paths in model_info.items():
        model_path = paths["model"]
        encoder_path = paths["encoder"]

        # Load model
        if not os.path.exists(model_path):
            text.insert(END, f"❌ {label_name} model not found.\n")
            continue
        model = load_model(model_path)

        # Load label encoder
        if not os.path.exists(encoder_path):
            text.insert(END, f"❌ {label_name} label encoder not found.\n")
            continue
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)

        # Predict
        preds = model.predict(features_reshaped)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_class = le.inverse_transform([pred_idx])[0]
        confidence = preds[0][pred_idx] * 100

        predictions[label_name] = (pred_class, confidence)
        text.insert(END, f"✅ Predicted {label_name} Type 🩺: {pred_class} ({confidence:.2f}% confidence)\n")

    # --- Plot waveform with predictions ---
    y, sr = librosa.load(filename, sr=None)
    plt.figure()
    librosa.display.waveshow(y, sr=sr, color='steelblue')
    plt.title("Waveform of Test Audio", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Save temporary waveform image
    waveplot_path = "results/test_waveplot.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(waveplot_path, bbox_inches='tight')
    plt.close()

    # Load the saved image
    waveplot_img = cv2.imread(waveplot_path)

    # Add predictions to the waveform image
    y_offset = 50
    for i, (label_name, (pred_class, confidence)) in enumerate(predictions.items()):
        text_overlay = f"{label_name} Prediction: {pred_class} ({confidence:.1f}%)"
        cv2.putText(
            waveplot_img, text_overlay,
            (50, y_offset + i * 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 0, 255) if label_name == "Heart" else (0, 128, 0),
            2, cv2.LINE_AA
        )

    # Display the annotated waveform
    cv2.imshow("Waveform with Heart & Lung Predictions", waveplot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: save annotated version
    annotated_path = "results/predicted_waveplot.png"
    cv2.imwrite(annotated_path, waveplot_img)
    text.insert(END, f"\n🩺 Combined Heart and Lung predictions visualized and saved to: {annotated_path}\n")
    
def close():
    main.destroy()


import tkinter as tk
from PIL import Image, ImageTk
from pymongo import MongoClient
import hashlib

# ------------------ MongoDB Connection ------------------

def connect_mongo():
    """Connect to MongoDB and return the users collection."""
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["soundauth"]          # Database name
        return db["users"]                # Collection name
    except Exception as e:
        messagebox.showerror("Error", f"MongoDB Connection Failed: {e}")
        return None


# ------------------ Security ------------------

def hash_password(password):
    """Securely hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


# ------------------ Signup Function ------------------

def signup(role):
    def register_user():
        username = username_entry.get().strip()
        password = password_entry.get().strip()

        if username and password:
            users = connect_mongo()
            if users is None:
                return

            existing_user = users.find_one({"username": username})
            if existing_user:
                messagebox.showerror("Error", "User already exists!")
                return

            hashed_pw = hash_password(password)
            users.insert_one({
                "username": username,
                "password": hashed_pw,
                "role": role
            })
            messagebox.showinfo("Success", f"{role} Signup Successful!")
            signup_window.destroy()
        else:
            messagebox.showerror("Error", "Please fill all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)


# ------------------ Login Function ------------------

def login(role):
    def verify_user():
        username = username_entry.get().strip()
        password = password_entry.get().strip()

        if username and password:
            users = connect_mongo()
            if users is None:
                return

            user = users.find_one({"username": username})
            if not user:
                messagebox.showerror("Error", "User not found!")
                return

            hashed_pw = hash_password(password)
            if user["password"] == hashed_pw and user["role"] == role:
                messagebox.showinfo("Success", f"{role} Login Successful!")
                login_window.destroy()
                if role == "Admin":
                    show_admin_buttons()
                else:
                    show_user_buttons()
            else:
                messagebox.showerror("Error", "Invalid credentials or role mismatch!")
        else:
            messagebox.showerror("Error", "Please fill all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk



# =========================
# Main Window
# =========================
main = tk.Tk()
main.title("GUI")

screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# =========================
# Background
# =========================
bg_image = Image.open("background.png")
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)

tk.Label(main, image=bg_photo)\
    .place(relx=0, rely=0, relwidth=1, relheight=1)

# =========================
# Fonts & Button Size
# =========================
TITLE_FONT = ('times', 18, 'bold')
BTN_FONT = ('times', 13, 'bold')
LOGIN_FONT = ("Helvetica", 11, "bold")

BTN_W = 0.18
BTN_W1 = 0.15

BTN_H = 0.055

# =========================
# Clear Buttons (ONLY buttons)
# =========================
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button):
            widget.destroy()

# =========================
# Login Screen (HOME)
# =========================
def show_login_screen():
    clear_buttons()

    tk.Button(main, text="Admin Signup", font=LOGIN_FONT,
              bg="lightblue",
              command=lambda: signup("Admin"))\
        .place(relx=0.02, rely=0.50, relwidth=BTN_W1, relheight=BTN_H)

    tk.Button(main, text="User Signup", font=LOGIN_FONT,
              bg="lightblue",
              command=lambda: signup("User"))\
        .place(relx=0.02, rely=0.58, relwidth=BTN_W1, relheight=BTN_H)

    tk.Button(main, text="Admin Login", font=LOGIN_FONT,
              bg="lightblue",
              command=lambda: login("Admin"))\
        .place(relx=0.02, rely=0.66, relwidth=BTN_W1, relheight=BTN_H)

    tk.Button(main, text="User Login", font=LOGIN_FONT,
              bg="lightblue",
              command=lambda: login("User"))\
        .place(relx=0.02, rely=0.74, relwidth=BTN_W1, relheight=BTN_H)

    tk.Button(main, text="Exit", font=LOGIN_FONT,
              bg="red",
              command=close)\
        .place(relx=0.02, rely=0.90, relwidth=BTN_W1, relheight=BTN_H)

# =========================
# Admin Dashboard
# =========================
def show_admin_buttons():
    clear_buttons()

    admin_buttons = [
        ("Upload Dataset", Upload_Dataset),
        ("Preprocess Dataset", Preprocess_Dataset),
        ("Train Test Splitting", Train_Test_Splitting),
        ("EDA", EDA),
        ("Build QDA", existing_classifier1),
        ("Build GB", existing_classifier2),
        ("Build GNB", existing_classifier3),
        ("Build LR", existing_classifier4),
        ("Build BiCNN-BiGRU", bicnn_bigru_classifier),
    ]

    start_y = 0.20
    gap = 0.06

    for i, (txt, cmd) in enumerate(admin_buttons):
        tk.Button(main, text=txt, font=BTN_FONT,
                  bg="lightpink",
                  command=cmd)\
            .place(relx=0.02,
                   rely=start_y + i * gap,
                   relwidth=BTN_W,
                   relheight=BTN_H)

    tk.Button(main, text="Logout", font=BTN_FONT,
              bg="red",
              command=show_login_screen)\
        .place(relx=0.02, rely=0.80, relwidth=BTN_W, relheight=BTN_H)

# =========================
# User Dashboard
# =========================
def show_user_buttons():
    clear_buttons()

    tk.Button(main, text="Prediction", font=BTN_FONT,
              bg="lightpink",
              command=Prediction)\
        .place(relx=0.02, rely=0.65, relwidth=BTN_W, relheight=BTN_H)

    tk.Button(main, text="Logout", font=BTN_FONT,
              bg="red",
              command=show_login_screen)\
        .place(relx=0.02, rely=0.73, relwidth=BTN_W, relheight=BTN_H)

# =========================
# Title
# =========================
Label(
    main,
    text="Automated Cardiorespiratory Sound Analysis for Disease Screening Using HLS-CMDS",
    bg="sky blue",
    fg="black",
    font=TITLE_FONT,
    height=2,
    width=120
).place(relx=0.5, rely=0.02, anchor='n')

# =========================
# Output Text Box 
# =========================
font_text = ('times', 12, 'bold')

text = Text(main, height=50, width=60, font=font_text)
scroll = Scrollbar(main, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.place(relx=0.50, rely=0.10, relwidth=0.48, relheight=0.85)

main.config(bg='SeaGreen1')

# =========================
# Start Application
# =========================
show_login_screen()
main.mainloop()
