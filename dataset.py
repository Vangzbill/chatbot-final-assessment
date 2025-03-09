import pandas as pd

data = {
    'question': [
        "Bagaimana cara berlangganan paket internet satelit bulanan?",
        "Berapa harga paket internet satelit?",
        "Apa syarat berlangganan internet satelit?",
        "Dimana saya bisa mendaftar internet satelit?",
        "Berapa lama proses pemasangan internet satelit?",
        "Apa yang harus dilakukan jika internet satelit bermasalah?",
        "Bagaimana cara membatalkan langganan internet satelit?",
        "Apakah ada biaya pemasangan internet satelit?",
        "Berapa kecepatan internet satelit yang tersedia?",
        "Apakah internet satelit bisa digunakan saat hujan?"
    ],
    'context': [
        "Untuk berlangganan paket internet satelit bulanan, Anda perlu mengunjungi situs web kami dan mendaftar dengan mengisi formulir online. Setelah itu, tim kami akan menghubungi Anda untuk konfirmasi.",
        "Harga paket internet satelit dimulai dari Rp 500.000 per bulan untuk paket basic dengan kecepatan 10 Mbps, hingga Rp 2.000.000 untuk paket premium dengan kecepatan 50 Mbps.",
        "Syarat berlangganan internet satelit meliputi KTP, alamat pemasangan yang valid, dan pembayaran biaya instalasi awal sebesar Rp 2.500.000.",
        "Pendaftaran internet satelit bisa dilakukan melalui website resmi kami di www.contoh.com atau mengunjungi kantor cabang terdekat di kota Anda.",
        "Proses pemasangan internet satelit membutuhkan waktu 2-3 hari kerja setelah survei lokasi. Survei lokasi dilakukan 1 hari setelah pendaftaran disetujui.",
        "Jika mengalami masalah dengan internet satelit, pelanggan dapat menghubungi call center 24 jam di nomor 14045 atau membuka tiket keluhan melalui aplikasi.",
        "Untuk membatalkan langganan, pelanggan harus memberikan pemberitahuan tertulis 30 hari sebelumnya dan melunasi semua tagihan yang tersisa.",
        "Biaya pemasangan internet satelit adalah Rp 2.500.000 untuk perangkat dan instalasi. Biaya ini dibayarkan sekali di awal berlangganan.",
        "Kami menyediakan beberapa pilihan kecepatan internet satelit mulai dari 10 Mbps, 25 Mbps, dan 50 Mbps tergantung paket yang dipilih.",
        "Internet satelit tetap bisa digunakan saat hujan ringan, namun kualitas sinyal mungkin berkurang saat hujan lebat atau badai."
    ],
    'split': ['train', 'train', 'train', 'train', 'train', 'train', 'test', 'test', 'test', 'test']
}

df = pd.DataFrame(data)

df.to_csv('sampel.csv', index=False)

print("Sample data saved to 'sampel.csv'. First few rows:")
print(df.head(3))