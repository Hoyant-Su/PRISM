import qrcode


arxiv_url = "https://arxiv.org/abs/2508.19325"
github_url = "https://github.com/Hoyant-Su/PRISM"

def generate_qrcode(url, qrcode_name):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(f"{qrcode_name}.png")

if __name__ == "__main__":
    generate_qrcode(arxiv_url, "arxiv_qrcode")
    generate_qrcode(github_url, "github_qrcode")