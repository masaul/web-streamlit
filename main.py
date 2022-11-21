import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pickle

page1, page2, page3, page4, page5 = st.tabs(["Home", "Data", "Preprocessing", "Input Model", "Prediksi"])

with page1:
    st.title("Klasifikasi Kredit Score Menggunakan Metode Naive Bayes Gaussian")
    st.write("Dataset Yang digunakan adalah **Kredit Score** dari Github: https://github.com/masaul/data-csv/blob/main/credit_score.csv")
    st.write("Link repository Github : [https://github.com/masaul/web-streamlit.git](https://github.com/masaul/web-streamlit.git) ")
    st.header("Deskripsi Data")
    st.write("Dataset yang digunakan memiliki 7 kolom dan juga memiliki type yang berbeda-beda. Untuk detail fitur ada di bawah ini:")
    st.markdown("""
        <ul>
            <li>
                Kolom 1: kode_kontrak
                <p> Kolom kode_kontrak merupakan identitas dari sebuah data. identitas data tidak perlu diikutkan untuk klasifikasi. Sehingga kolom ini akan diabaikan saat klasifikasi data. </p>
            </li>
            <li>
                Kolom 2: pendapatan_setahun_juta
                <p> Kolom pendapatan_setahun_juta merupakan data yang bertype numerik. </p>
            </li>
            <li>
                Kolom 3: kpr_aktif
                <p> Kolom kpr_aktif merupakan data bertype categorial sehingga harus di normalisasikan agar berubah menjadi numerik. </p>
            </li>
            <li>
                Kolom 4: durasi_pinjaman_bulan
                <p> Kolom durasi_pinjaman_bulan data bertype numerik. </p>
            </li>
            <li>
                Kolom 5: jumlah_tanggungan
                <p> Kolom jumlah_tanggungan merupakan data bertype numerik </p>
            </li>
            <li>
                Kolom 6: rata_rata_overdue
                <p> Kolom rata_rata_overdue merupakan data bertype categorial sehingga harus di normalisasikan agar berubah menjadi numerik </p>
            </li>
            <li>
                Kolom 7: risk_rating
                <p> Kolom risk_rating merupakan class dengan type data numerik. </p>
            </li>
        </ul>
    """, unsafe_allow_html=True)

with page2:
    st.title("Dataset Kredit Score")
    data = pd.read_csv("https://raw.githubusercontent.com/masaul/data-csv/main/credit_score.csv")
    deleteCol = data.drop(["Unnamed: 0"], axis=1)
    st.write(deleteCol)

with page3:
    st.title("Halaman PreProcessing")
    st.write("Preprocessing data merupakan tahapan untuk melakukan mining data sebelum tahap pemrosesan. fungsi preprocessing data untuk mengubah data mentah menjadi data yang mudah dipahami.")
    st.markdown("""
        <ol>
            <li>Data Cleaning</li>
            <li>Transformasi Data</li>
            <li>Mengurangi Data</li>
        </ol>
    """, unsafe_allow_html=True)
    st.write("Disini preprocessing menggunakan transformasi data dengan metode MinMaxScaller()")
    st.write("MinMaxScaler() merupakan transformasi data dengan rentang tertentu, rentang yang digunakan disini yaitu 0 - 1. Rumus transformasi data dapat menggunakan berikut:")
    rumus1 = '''X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))'''
    rumus2 = '''X_scaled = X_std * (max - min) + min'''
    st.code(rumus1, language="python")
    st.code(rumus2, language="python")
    st.write("Sebelum itu harus di encoder dulu data attribut yang memiliki type categorial harus diubah menjadi type numerik agar bisa dilakukan preprocessing.")

    st.subheader("Split Data")
    codeSplit = '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)'''
    st.code(codeSplit, language="python")
    st.write("Split Data disini digunakan untuk memisahkan data menjadi nilai acak. data yang digunakan sebagai data testing sebesar 0.3 atau 30% dari data yang digunakan. sehingga sisanya digunakan sebagai data training.")



    creditScoreRaw = pd.read_csv("https://raw.githubusercontent.com/masaul/data-csv/main/credit_score.csv")


    dataCreditScore_withoutColumns= pd.DataFrame(creditScoreRaw, columns=['kode_kontrak','pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan','risk_rating'])
    # fiturWithout = pd.DataFrame = pd.DataFrame(fitur, columns=['Pendapatan Setahun (Juta)', 'Jumlah Tanggungan', 'Durasi Pinjaman (Bulan)'])

    # Encode Data menjadi numerik
    encodeAvarage = pd.get_dummies(creditScoreRaw['rata_rata_overdue'])
    encodeKprAktif=pd.get_dummies(creditScoreRaw['kpr_aktif'])

    # encodeFiturAvarage = pd.get_dummies(fitur['Rata-rata     Overdue'])
    # encodeFiturKpr = pd.get_dummies(fitur['KPR Aktif'])

    # Menggabungkan data yang sudah di encode
    concatCreditScoreRaw = pd.concat([dataCreditScore_withoutColumns, encodeKprAktif, encodeAvarage], axis=1)
    # concatFitur = pd.concat([fiturWithout, encodeFiturAvarage, encodeFiturKpr])


    dataframeRiskRating = pd.DataFrame(creditScoreRaw, columns=['risk_rating'])

    # Menghapus class risk rating
    dropRiskRating = concatCreditScoreRaw.drop(['risk_rating'], axis=1)

    # Menggabungkan class
    concatCreditScoreRaw2 = pd.concat([dropRiskRating, dataframeRiskRating], axis=1)


    # Preprocessing
    preprocessingData = pd.DataFrame(creditScoreRaw, columns=['pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan'])
    preprocessingData.to_numpy()
    scaler = MinMaxScaler()
    resultPreprocessingData = scaler.fit_transform(preprocessingData)
    resultPreprocessingData = pd.DataFrame(resultPreprocessingData, columns=['pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan'])

    dropColumnPreprocessingData = concatCreditScoreRaw2.drop(['pendapatan_setahun_juta','durasi_pinjaman_bulan','jumlah_tanggungan'], axis=1)
    concatCreditScoreRaw3 = pd.concat([dropColumnPreprocessingData, resultPreprocessingData], axis=1)

    dropColumn_RiskRatingPre = concatCreditScoreRaw3.drop(['risk_rating'], axis=1)
    resultData = pd.concat([dropColumn_RiskRatingPre, dataframeRiskRating], axis=1)

    st.subheader("Data Sesudah di Processing")
    st.write(resultData)

    X = resultData.iloc[:,1:11].values
    y = resultData.iloc[:,11].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    akurasi = accuracy_score(y_test,Y_pred)

    st.subheader("Hasil Akurasi")
    st.write("Didapatkan dari data training 70% dan Data Testing 30%")
    st.write(akurasi)

with page4:
    st.title("Input Data Model")


    # membuat input
    pendapatan_setahun = st.text_input("Pendapatan Setahun(juta)")
    kpr = st.radio("KPR", ("aktif", "tidak aktif"))
    jumlah_tanggungan = st.text_input("Jumlah Tanggungan")
    durasi_pinjaman = st.selectbox("Durasi (bulan)", ("12", "24", "36", "48"))
    overdue = st.selectbox("Overdue", ("0 - 30 days", "31 - 45 days", "46 - 60 days", "61 - 90 days", "> 90 days"))


    # section output
    def submit():
        # cek input 
        # scaler = MinMaxScaler()
        # scaler.fit([[int(pendapatan_setahun),int(durasi_pinjaman), int(jumlah_tanggungan)]])
        # MinMaxScaler()
        # filename = 'scaler.save'
        # joblib.dump(scaler, open(filename, 'wb'))

        scaler = joblib.load("scaler.save")
        normalize = scaler.transform([[int(pendapatan_setahun),int(durasi_pinjaman), int(jumlah_tanggungan)]])[0].tolist()

        kpr_ya = 0
        kpr_tidak = 0
        if kpr == "aktif":
            kpr_ya = 1
        else:
            kpr_tidak = 1

        overdues = [0,0,0,0,0]
        if(overdue == "0 - 30 days"):
            overdues[0] = 1
        elif(overdue == "31 - 45 days"):
            overdues[1] = 1
        elif(overdue == "46 - 60 days"):
            overdues[2] = 1
        elif(overdue == "61 - 90 days"):
            overdues[3] = 1
        else:
            overdues[4] = 1

        # create data input
        data_input = {
            "pendapatan_setahun_juta" : normalize[0],
            "durasi_pinjaman_bulan" : normalize[1],
            "jumlah_tanggungan" : normalize[2],
            "overdue_0 - 30 days": overdues[0],
            "overdue_31 - 45 days": overdues[1],
            "overdue_46 - 60 days": overdues[2],
            "overdue_61 - 90 days": overdues[3],
            "overdue_> 90 days": overdues[4],
            "KPR_TIDAK" : kpr_tidak,
            "KPR_YA": kpr_ya
        }

        inputs = np.array([[val for val in data_input.values()]])

        # filenameModel = "model.joblib"
        # joblib.dump(naive_bayes_classifier, filename)

        model = joblib.load("model.joblib")
        # with open("model.sav", "rb") as model_buffer:
            # model = pickle.load(model_buffer)
        pred = model.predict(inputs)

        return pred
        


    # create button submit
    submitted = st.button("Prediksi")
    if submitted:
        st.text("Hasil prediksi ada pada halaman Prediksi")
        with page5:
            st.write("Hasil prediksi risk rating yang di peroleh yaitu:")
            st.text(submit())

with page5:
    if not submitted:
        st.write("Belum ada prediksi, Harap masukkan input model dahulu!")



