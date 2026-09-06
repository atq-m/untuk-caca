import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import textwrap
import random
import time
from collections import deque
import os
import sys


def resource_path(relative_path):
    """
    Mengambil path resource baik saat program dijalankan
    sebagai source code maupun sebagai aplikasi PyInstaller.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# =========================================================
# 1. MEMBUKA KAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Kamera tidak dapat dibuka.")
    exit()


# =========================================================
# 2. WINDOW
# =========================================================

window_name = "Untuk Caca"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)


# =========================================================
# 3. MEMUAT FONT
# =========================================================

title_font = ImageFont.truetype(
    resource_path("fonts/title.ttf"),
    30
)

text_font = ImageFont.truetype(
    resource_path("fonts/text.ttf"),
    17
)

accent_font = ImageFont.truetype(
    resource_path("fonts/accent.ttf"),
    28
)

small_font = ImageFont.truetype(
    resource_path("fonts/text.ttf"),
    13
)

answer_font = ImageFont.truetype(
    resource_path("fonts/text.ttf"),
    22
)

answer_title_font = ImageFont.truetype(
    resource_path("fonts/accent.ttf"),
    42
)


# =========================================================
# 4. MEMUAT HAND LANDMARKER
# =========================================================

hand_base_options = python.BaseOptions(
    model_asset_path=resource_path("models/hand_landmarker.task"
))

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)


# =========================================================
# 5. MEMUAT FACE LANDMARKER
# =========================================================

face_base_options = python.BaseOptions(
    model_asset_path=resource_path("models/face_landmarker.task"
))

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)


# =========================================================
# 6. SEMUA PESAN
# =========================================================

messages = {

    0: """
Ada 5 pesan yang mau aku sampaikan ke kamu.

Kamu bisa nunjukin jari kamu dari 1-5
buat ngebacanya.
""",

    1: """
Halo Reva Audisya, atau akrab aku panggil Caca.
""",

    2: """
Aku ga nyangka banget bisa kenal dan deket sama orang kayak kamu.

Orang yang cantik, baik terlebih lagi bagiku kamu punya pemikiran yang dewasa di usia muda.

Terima kasih udah hadir di hidup aku, kamu bener-bener mewarnai hidup aku yang dulu
abu,
hampa,
dan yaa...
gitu gitu aja.
""",

    3: """
Tapi setelah ketemu kamu, hidup aku bisa lebih baik.

Aku yang dulu sangat pemalas, sekarang udah berusaha untuk lebih rajin,

Aku yang dulu cukup jauh dari agama, sekarang aku sadar dan lebih mendekatkan diri ke Sang Pencipta,

Dan hal terpenting yang aku rasa dengan adanya kamu, aku paham bahwa aku bukan orang gagal.

Aku adalah orang yang sedang berproses dan memang mungkin butuh waktu lebih lama dari orang lain.

Hidupku jauh lebih semangat, jauh lebih termotivasi, dan jauh lebih berambisi.

Dan semua itu karena kamu, Ca.
""",

    4: """
Maaf kamu ketemu aku ketika aku masih kayak gini, masih berproses,
belum punya apa-apa,
masih banyak gagalnya,
masih banyak ngeluhnya.

Kita bener-bener terpaut umur yang lumayan jauh, dan kalo boleh jujur aku suka mikir

'Boleh ga ya aku yang sekarang kayak gini, belum jadi apa-apa dan umur kita yang terpaut cukup jauh...
menjalin sebuah keterikatan, komitmen dan juga hubungan yang bukan sekedar dekat tanpa tujuan.'

Tapi aku juga gamau menyesal di kemudian hari, bahwa aku pernah menyia-nyiakan orang sebaik, secantik dan seberharga kamu.

Aku janji kedepannya aku bakal berusaha lebih keras, buat menggapai mimpi aku, dan juga memantaskan diri untuk lebih layak berada di samping kamu, sampai orang-orang gaada yang mempertanyakan hal-hal aneh terkait kita berdua.

Di hari ini, aku beranikan diri buat menyampaikan apa-apa yang selama ini aku pikirin.

Semua ini akan berakhir pada keputusan kamu.
""",

    5: """
Caa...

Aku suka sama kamu, Aku sayang sama kamu.

Apakah sekiranya kamu memperbolehkan aku untuk
terus memperjuangkan mimipiku di sisimu,
terus mendapatkan supportmu di sela-sela saat aku buntu,
dan terus berusaha melayakkan diriku sebagai pasanganmu.

Aku menyampaikan ini bukan karena aku ingin merebut kebebasan kamu,
merenggut waktu kamu ataupun membatasi segala kegiatan kamu.

Aku menyampaikan ini karena

Aku ingin menunjukkan keseriusanku,
Aku ingin memastikan apakah aku masih diizinkan untuk berada di samping kamu,
Dan aku ingin selalu ada dibagaimanapun kondisi kamu.

Aku tau ini egois, tapi caa...

Maukah kamu jadi pacar aku?

Kamu bisa nganggukin kepala kamu kalo kamu mau,
atau kamu pun bisa menggelengkan kepala kalo kamu ga bersedia.
"""
}


# =========================================================
# 7. MEMBAGI PESAN MENJADI HALAMAN
# =========================================================

def create_pages(
    message,
    max_lines=5,
    chars_per_line=44
):

    all_lines = []

    paragraphs = message.strip().split("\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph == "":

            all_lines.append("")

        else:

            wrapped_lines = textwrap.wrap(
                paragraph,
                width=chars_per_line
            )

            all_lines.extend(
                wrapped_lines
            )


    pages = []

    for i in range(
        0,
        len(all_lines),
        max_lines
    ):

        page = all_lines[
            i:i + max_lines
        ]

        pages.append(page)


    return pages


# =========================================================
# 8. MEMBUAT HALAMAN
# =========================================================

pages = {}

for number, message in messages.items():

    pages[number] = create_pages(
        message
    )


# =========================================================
# 9. FUNGSI GAMBAR HATI
# =========================================================

def draw_heart(
    draw,
    x,
    y,
    size,
    color
):

    radius = size // 2


    # Dua bagian atas hati

    draw.ellipse(

        (
            x,
            y,

            x + radius,
            y + radius
        ),

        fill=color
    )


    draw.ellipse(

        (
            x + radius,
            y,

            x + size,
            y + radius
        ),

        fill=color
    )


    # Bagian bawah hati

    draw.polygon(

        [

            (
                x,
                y + radius // 2
            ),

            (
                x + size,
                y + radius // 2
            ),

            (
                x + size // 2,
                y + size
            )

        ],

        fill=color
    )


# =========================================================
# 10. MEMBUAT LOVE ANIMATION
# =========================================================

loves = []


def create_loves(
    width,
    height,
    amount=25
):

    global loves

    loves = []


    for i in range(amount):

        loves.append({

            "x": random.randint(
                20,
                width - 40
            ),

            "y": random.randint(
                height,
                height + 500
            ),

            "size": random.randint(
                12,
                28
            ),

            "speed": random.uniform(
                1.0,
                3.5
            ),

            "color": (

                random.randint(220, 255),

                random.randint(80, 170),

                random.randint(100, 200)

            )

        })


# =========================================================
# 11. FUNGSI UPDATE LOVE
# =========================================================

def update_loves(
    draw,
    width,
    height
):

    global loves


    for love in loves:


        # Bergerak ke atas

        love["y"] -= love["speed"]


        # Gambar hati

        draw_heart(

            draw,

            int(love["x"]),

            int(love["y"]),

            love["size"],

            love["color"]

        )


        # Jika keluar layar

        if love["y"] < -50:


            love["y"] = random.randint(
                height,
                height + 300
            )


            love["x"] = random.randint(
                20,
                width - 40
            )


# =========================================================
# 12. GAMBAR PESAN UTAMA
# =========================================================

def draw_message_card(
    frame,
    finger_count,
    page_number
):

    height, width = frame.shape[:2]


    card_left = 20

    card_right = width - 20

    card_top = int(height * 0.50)

    card_bottom = height - 15


    # =====================================================
    # GLASS EFFECT
    # =====================================================

    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (
            card_left,
            card_top
        ),

        (
            card_right,
            card_bottom
        ),

        (
            20,
            20,
            20
        ),

        -1
    )


    frame = cv2.addWeighted(

        overlay,

        0.78,

        frame,

        0.22,

        0

    )


    # =====================================================
    # BORDER
    # =====================================================

    cv2.rectangle(

        frame,

        (
            card_left,
            card_top
        ),

        (
            card_right,
            card_bottom
        ),

        (
            230,
            230,
            230
        ),

        1
    )


    # =====================================================
    # OPENCV → PIL
    # =====================================================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(
        frame_rgb
    )


    draw = ImageDraw.Draw(
        image
    )


    # =====================================================
    # TITLE
    # =====================================================

    if finger_count == 0:

        title = "Untuk Caca"

    else:

        title = f"Pesan {finger_count}"


    title_box = draw.textbbox(

        (0, 0),

        title,

        font=title_font
    )


    title_width = (
        title_box[2]
        - title_box[0]
    )


    title_x = (
        width - title_width
    ) // 2


    draw.text(

        (
            title_x,
            card_top + 8
        ),

        title,

        font=title_font,

        fill=(
            255,
            255,
            255
        )

    )


    # =====================================================
    # ORNAMEN
    # =====================================================

    ornament = "♡  ─── ✦ ───  ♡"


    ornament_box = draw.textbbox(

        (0, 0),

        ornament,

        font=small_font
    )


    ornament_width = (

        ornament_box[2]

        - ornament_box[0]

    )


    ornament_x = (

        width - ornament_width

    ) // 2


    draw.text(

        (
            ornament_x,
            card_top + 47
        ),

        ornament,

        font=small_font,

        fill=(
            210,
            210,
            210
        )

    )


    # =====================================================
    # PESAN
    # =====================================================

    current_page = pages[
        finger_count
    ][page_number]


    total_pages = len(
        pages[finger_count]
    )


    y = card_top + 75


    line_spacing = 20


    for line in current_page:


        if line == "":

            y += 8


        else:


            if line.lower().startswith("caa"):


                draw.text(

                    (
                        card_left + 20,
                        y - 5
                    ),

                    line,

                    font=accent_font,

                    fill=(
                        255,
                        235,
                        240
                    )

                )


                y += 30


            else:


                draw.text(

                    (
                        card_left + 20,
                        y
                    ),

                    line,

                    font=text_font,

                    fill=(
                        245,
                        245,
                        245
                    )

                )


                y += line_spacing


    # =====================================================
    # FOOTER
    # =====================================================

    footer_y = card_bottom - 25


    page_text = (
        f"{page_number + 1}"
        f" / "
        f"{total_pages}"
    )


    page_box = draw.textbbox(

        (0, 0),

        page_text,

        font=small_font
    )


    page_width = (
        page_box[2]
        - page_box[0]
    )


    draw.text(

        (
            card_right
            - page_width
            - 15,

            footer_y
        ),

        page_text,

        font=small_font,

        fill=(
            200,
            200,
            200
        )

    )


    hint = "N: lanjut   B: kembali   Q: keluar"


    hint_box = draw.textbbox(

        (0, 0),

        hint,

        font=small_font
    )


    hint_width = (
        hint_box[2]
        - hint_box[0]
    )


    hint_x = (
        width - hint_width
    ) // 2


    draw.text(

        (
            hint_x,
            footer_y
        ),

        hint,

        font=small_font,

        fill=(
            160,
            160,
            160
        )

    )


    frame = cv2.cvtColor(

        np.array(image),

        cv2.COLOR_RGB2BGR

    )


    return frame


# =========================================================
# 13. GAMBAR JAWABAN "IYA"
# =========================================================

def draw_yes_screen(
    frame
):

    height, width = frame.shape[:2]


    overlay = frame.copy()


    overlay[:, :] = (
        40,
        130,
        70
    )


    frame = cv2.addWeighted(

        overlay,

        0.72,

        frame,

        0.28,

        0

    )


    frame_rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    image = Image.fromarray(
        frame_rgb
    )


    draw = ImageDraw.Draw(
        image
    )


    update_loves(
        draw,
        width,
        height
    )


    title = "YAYY ♡"


    title_box = draw.textbbox(

        (0, 0),

        title,

        font=answer_title_font
    )


    title_width = (
        title_box[2]
        - title_box[0]
    )


    draw.text(

        (
            (width - title_width) // 2,
            80
        ),

        title,

        font=answer_title_font,

        fill=(
            255,
            255,
            255
        )

    )


    message = """

Terimakasih sayang,

maaf baru ngungkapin sekarang
dan tidak mengungkapkannya
secara langsung.

Nanti ketika aku udah kerja
aku janji bakal datangin kamu
ke Jakarta yaa!

"""


    lines = []


    for paragraph in message.strip().split("\n"):


        paragraph = paragraph.strip()


        if paragraph == "":

            lines.append("")

        else:

            wrapped = textwrap.wrap(
                paragraph,
                width=34
            )


            lines.extend(
                wrapped
            )


    y = 170


    for line in lines:


        if line == "":

            y += 18


        else:


            line_box = draw.textbbox(

                (0, 0),

                line,

                font=answer_font
            )


            line_width = (
                line_box[2]
                - line_box[0]
            )


            draw.text(

                (
                    (width - line_width) // 2,
                    y
                ),

                line,

                font=answer_font,

                fill=(
                    255,
                    255,
                    255
                )

            )


            y += 32


    frame = cv2.cvtColor(

        np.array(image),

        cv2.COLOR_RGB2BGR

    )


    return frame


# =========================================================
# 14. GAMBAR JAWABAN "TIDAK"
# =========================================================

def draw_no_screen(
    frame
):

    height, width = frame.shape[:2]


    overlay = frame.copy()


    overlay[:, :] = (
        40,
        0,
        100
    )


    frame = cv2.addWeighted(

        overlay,

        0.72,

        frame,

        0.28,

        0

    )


    frame_rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    image = Image.fromarray(
        frame_rgb
    )


    draw = ImageDraw.Draw(
        image
    )


    title = "It's Okayy Caa"


    title_box = draw.textbbox(

        (0, 0),

        title,

        font=answer_title_font
    )


    title_width = (
        title_box[2]
        - title_box[0]
    )


    draw.text(

        (
            (width - title_width) // 2,
            height // 2 - 70
        ),

        title,

        font=answer_title_font,

        fill=(
            255,
            255,
            255
        )

    )


    message = """

Makasih yaa udah jujur
sama perasaan kamu ke aku.

"""


    y = height // 2 + 20


    for line in message.strip().split("\n"):


        line = line.strip()


        if line == "":

            y += 10

            continue


        line_box = draw.textbbox(

            (0, 0),

            line,

            font=answer_font
        )


        line_width = (
            line_box[2]
            - line_box[0]
        )


        draw.text(

            (
                (width - line_width) // 2,
                y
            ),

            line,

            font=answer_font,

            fill=(
                245,
                245,
                245
            )

        )


        y += 35


    frame = cv2.cvtColor(

        np.array(image),

        cv2.COLOR_RGB2BGR

    )


    return frame


# =========================================================
# 15. STATUS PROGRAM
# =========================================================

current_message = 0

current_page = 0

last_finger_count = -1


# Mode:
# message = membaca pesan
# yes = jawaban angguk
# no = jawaban geleng

mode = "message"


# =========================================================
# 16. VARIABEL DETEKSI GERAK KEPALA
# =========================================================

nose_y_history = deque(
    maxlen=12
)

nose_x_history = deque(
    maxlen=12
)


last_answer_time = 0

cooldown = 2.0


# Threshold

NOD_THRESHOLD = 0.15

SHAKE_THRESHOLD = 0.15


# =========================================================
# 17. LOOP UTAMA
# =========================================================

while True:


    success, frame = cap.read()


    if not success:

        break


    height, width = frame.shape[:2]


    # =====================================================
    # CONVERT KE RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame

    )


    # =====================================================
    # HAND DETECTION
    # =====================================================

    hand_result = hand_detector.detect(
        mp_image
    )


    if hand_result.hand_landmarks:


        hand_landmarks = (
            hand_result.hand_landmarks[0]
        )


        # TELUNJUK

        if (
            hand_landmarks[8].y
            <
            hand_landmarks[6].y
        ):

            index = 1

        else:

            index = 0


        # JARI TENGAH

        if (
            hand_landmarks[12].y
            <
            hand_landmarks[10].y
        ):

            middle = 1

        else:

            middle = 0


        # JARI MANIS

        if (
            hand_landmarks[16].y
            <
            hand_landmarks[14].y
        ):

            ring = 1

        else:

            ring = 0


        # KELINGKING

        if (
            hand_landmarks[20].y
            <
            hand_landmarks[18].y
        ):

            pinky = 1

        else:

            pinky = 0


        # IBU JARI

        if (
            hand_landmarks[4].x
            >
            hand_landmarks[3].x
        ):

            thumb = 1

        else:

            thumb = 0


        finger_count = (

            thumb

            + index

            + middle

            + ring

            + pinky

        )


        if finger_count != last_finger_count:


            current_message = finger_count


            current_page = 0


            last_finger_count = finger_count


    # =====================================================
    # FACE DETECTION
    # HANYA AKTIF SAAT PESAN KE-5
    # =====================================================

    if (
        current_message == 5
        and mode == "message"
    ):


        face_result = face_detector.detect(
            mp_image
        )


        if face_result.face_landmarks:


            face_landmarks = (
                face_result.face_landmarks[0]
            )


            # Landmark hidung

            nose = face_landmarks[1]


            nose_x_history.append(
                nose.x
            )


            nose_y_history.append(
                nose.y
            )


            # =================================================
            # ANALISIS GERAK JIKA DATA SUDAH CUKUP
            # =================================================

            if (
                len(nose_y_history)
                >= 10
                and
                time.time()
                - last_answer_time
                > cooldown
            ):


                # ---------------------------------------------
                # ANALISIS ANGGUK
                # ---------------------------------------------

                y_values = list(
                    nose_y_history
                )


                y_range = (
                    max(y_values)
                    -
                    min(y_values)
                )


                # ---------------------------------------------
                # ANALISIS GELENG
                # ---------------------------------------------

                x_values = list(
                    nose_x_history
                )


                x_range = (

                    max(x_values)

                    -

                    min(x_values)

                )


                # =================================================
                # PRIORITAS GELOMBANG TERBESAR
                # =================================================

                if (

                    y_range > NOD_THRESHOLD

                    and

                    y_range > x_range

                ):


                    mode = "yes"


                    last_answer_time = time.time()


                    create_loves(
                        width,
                        height
                    )


                    nose_y_history.clear()

                    nose_x_history.clear()


                elif (

                    x_range > SHAKE_THRESHOLD

                    and

                    x_range > y_range

                ):


                    mode = "no"


                    last_answer_time = time.time()


                    nose_y_history.clear()

                    nose_x_history.clear()


    # =====================================================
    # MENAMPILKAN MODE
    # =====================================================

    if mode == "yes":


        frame = draw_yes_screen(
            frame
        )


    elif mode == "no":


        frame = draw_no_screen(
            frame
        )


    else:


        frame = draw_message_card(

            frame,

            current_message,

            current_page

        )


    # =====================================================
    # MENAMPILKAN FRAME
    # =====================================================

    cv2.imshow(

        window_name,

        frame

    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(
        10
    ) & 0xFF


    # Q = KELUAR

    if key == ord("q"):

        break


    # R = KEMBALI KE PESAN

    elif key == ord("r"):


        mode = "message"


        nose_x_history.clear()

        nose_y_history.clear()


    # N = HALAMAN BERIKUTNYA

    elif (
        key == ord("n")
        and mode == "message"
    ):


        total_pages = len(
            pages[current_message]
        )


        if current_page < total_pages - 1:

            current_page += 1


    # B = HALAMAN SEBELUMNYA

    elif (
        key == ord("b")
        and mode == "message"
    ):


        if current_page > 0:

            current_page -= 1


    # =====================================================
    # WINDOW DITUTUP
    # =====================================================

    if (

        cv2.getWindowProperty(

            window_name,

            cv2.WND_PROP_VISIBLE

        )

        < 1

    ):

        break


# =========================================================
# 18. MENUTUP PROGRAM
# =========================================================

cap.release()

hand_detector.close()

face_detector.close()

cv2.destroyAllWindows()

print("Program selesai.")