import os
from pathlib import Path
from captcha.image import ImageCaptcha
import string
import numpy as np
import threading
import logging

font_case_sensitive_dir = Path('data/fonts/case_sensitive')
font_non_case_sensitive_dir = Path('data/fonts/non_case_sensitive')
font_case_sensitive_paths = [os.path.join(font_case_sensitive_dir, fn) for fn in os.listdir(font_case_sensitive_dir)]
font_non_case_sensitive_paths = [os.path.join(font_non_case_sensitive_dir, fn) for fn in
                                 os.listdir(font_non_case_sensitive_dir)]

all_case_sensitive_chars = np.array(list(string.ascii_letters + string.digits))
all_non_case_sensitive_chars = np.array(list(string.ascii_uppercase + string.digits))


def gen_captcha(fonts: tuple, chars: tuple, n_captcha, output_dir, min_n_chars=5, max_n_chars=15):
    n = max_n_chars - min_n_chars + 1
    n_captcha_per_n = (n_captcha - 1) // (n * len(fonts)) + 1

    for f, c in zip(fonts, chars):
        n_chars = c.shape[0]
        for n_char in range(min_n_chars, max_n_chars + 1):
            img_captcha_gen = ImageCaptcha(width=n_char * 50, height=50, fonts=f, font_sizes=[50])

            i = 0
            while i < n_captcha_per_n:
                text = ''.join(
                    c[np.random.randint(0, n_chars, size=n_char)]).strip()
                img_captcha_gen.write(text, output=os.path.join(output_dir, f'{text}.png'))
                i += 1


def test_font(font: str):
    text = string.ascii_letters + string.digits
    font_name = font.split('/')[-1]
    img_captcha_gen = ImageCaptcha(width=len(text) * 15, fonts=[font])
    img_captcha_gen.write(text, f'data/test_images/{font_name}.png')


def gen_data(output_dir, n_captcha):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    n_threads = 10
    n_captcha_per_thread = (n_captcha - 1) // n_threads + 1

    logging.info(f'Creating {n_threads} threads')
    threads = [threading.Thread(target=gen_captcha, kwargs={
        'fonts': (font_case_sensitive_paths, font_non_case_sensitive_paths),
        'chars': (all_case_sensitive_chars, all_non_case_sensitive_chars),
        'n_captcha': n_captcha_per_thread,
        'min_n_chars': 5,
        'max_n_chars': 5,
        'output_dir': output_dir,
    }) for _ in range(n_threads)]

    for i, thread in enumerate(threads):
        logging.info(f'Starting thread {i}')
        thread.start()

    for i, thread in enumerate(threads):
        logging.info(f'Joining thread {i}')
        thread.join()


def test_data():
    for font in font_case_sensitive_paths:
        test_font(font)


if __name__ == '__main__':
    # test_data()
    gen_data(output_dir=Path('data/images'), n_captcha=5000)
    # gen_data(output_dir=Path('data/test_images'), n_captcha=1000)
