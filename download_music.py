# 출처: http://www.youtube.com/watch?v=BaW_jenozKc

from __future__ import unicode_literals
import youtube_dl
from bs4 import BeautifulSoup
import urllib
import requests
import os
import time

query_page = 'http://www.youtube.com/results?search_query='
mv_page = 'https://www.youtube.com'

ydl_opts = {
    'format': 'bestaudio/best',''
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}


def search_good_youtube_link(query):
    query = query.lower()                               # query를 소문자로 변형
    path = urllib.parse.quote_plus(query)               # string을 url 형태로 변환

    page = requests.get(query_page + path)               # youtube 검색 url 생성후 페이지 요청
    soup = BeautifulSoup(page.content, 'html.parser')
    html_page = soup.prettify()

    idx = html_page.lower().find('title="' + query)     # 먼저 쿼리와 관련된 제목을 찾는다.
    ridx = html_page.rfind("href=", 0, idx)             # 쿼리로 부터 'href='를 가진 문자의 위치를 거꾸로 찾는다

    if idx == -1:
        return ''

    page_str = ''
    ridx += 6
    while True:
        if html_page[ridx] == '"':
            break;
        page_str += html_page[ridx]
        ridx += 1

    return mv_page + page_str


def search_youtube_link(query):
    query = query.lower()                               # query를 소문자로 변형
    path = urllib.parse.quote_plus(query)               # string을 url 형태로 변환

    page = requests.get(query_page + path)               # youtube 검색 url 생성후 페이지 요청
    soup = BeautifulSoup(page.content, 'html.parser')
    html_page = soup.prettify()

    idx = html_page.lower().find('<h3 class="yt-lockup-title')     # 먼저 youtube 검색시 제목 표시란을 찾는다.
    new_idx = html_page.find("href=", idx)             # 쿼리로 부터 'href='를 가진 문자의 위치를 거꾸로 찾는다

    if idx == -1:
        return ''

    page_str = ''
    new_idx += 6
    while True:
        if html_page[new_idx] == '"':
            break;
        page_str += html_page[new_idx]
        new_idx += 1

    return mv_page + page_str


# 다운로드할 폴더명, 분할된 음악을 저장할 폴더, 음악 이미지를 저장할 폴더, 다운로드할 가수 및 노래 제목이 있는 폴더
def download_youtube_mp3(download_dir, file):
    # ydl_opts['outtmpl'] = r'D:\MyPythonProject\Youtube_DL\download_file\happy_music\%(title)s.%(ext)s'
    ydl_opts['outtmpl'] = os.path.join(download_dir, '%(title)s.%(ext)s')

    f_error = open(os.path.join(download_dir, 'error.txt'), 'w')

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        f = open(file)
        file_list = []
        idx = 0
        while True:
            line = f.readline()
            if not line:
                break

            line = line[:len(line) - 1]  # 맨뒤의 개행문자 삭제
            items = line.split('/')

            music_name = items[1] + ' - ' + items[0]
            print(music_name)
            try:
                link = search_youtube_link(music_name)

                if link == '':
                    continue
                file_list.append(link)

                ydl.download(file_list[idx:])

                time.sleep(10)
                idx += 1
            except Exception as e:
                print(str(e))
                f_error.write(items[0] + '/' + items[1] + '\n')       # error 난 음악 출력.
        f.close()
        # print(file_list)
        # ydl.download(filelist)
    f_error.close()


def download_youtube_mp3_2(download_dir, file):
    # ydl_opts['outtmpl'] = r'D:\MyPythonProject\Youtube_DL\download_file\happy_music\%(title)s.%(ext)s'

    f_error = open(os.path.join(download_dir, 'error.txt'), 'w')
    f = open(file)

    idx = 0
    music_len = 500

    while True:
        line = f.readline()
        if not line:
            break

        line = line[:len(line) - 1]  # 맨뒤의 개행문자 삭제
        items = line.split('/')

        music_name = items[1] + ' - ' + items[0]
        print(music_name)

        if idx % 100 == 0:
            print("Process {} / {} Done".format(idx, music_len))
        idx += 1

        try:
            link = search_youtube_link(music_name)

            if link == '':
                continue

            youtube_download_opts = {
                'format': 'bestaudio/best', ''
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                'outtmpl': os.path.join(download_dir, music_name) + '.%(ext)s'
            }

            with youtube_dl.YoutubeDL(youtube_download_opts) as ydl:
                ydl.download([link, ])

        except Exception as e:
            print(str(e))
            f_error.write(items[0] + '/' + items[1] + '\n')       # error 난 음악 출력.

    f.close()
    f_error.close()


# file에 있는 내용을 다운로드 한다.
def download_youtube_mp3_top_n(download_dir, file, top_n, dump_dir):
    # ydl_opts['outtmpl'] = r'D:\MyPythonProject\Youtube_DL\download_file\happy_music\%(title)s.%(ext)s'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    f_error = open(os.path.join(dump_dir, 'error.txt'), 'w')
    f_download = open(os.path.join(dump_dir, 'top_download_list.txt'), 'w')
    f = open(file)

    download_num = 0

    while True:
        line = f.readline()
        if not line:
            break

        line = line[:len(line) - 1]  # 맨뒤의 개행문자 삭제
        items = line.split('/')

        music_name = items[1] + ' - ' + items[0]
        print(music_name)

        if download_num % 10 == 0:
            print("Process {} / {} Done".format(download_num, top_n))

        try:
            link = search_good_youtube_link(music_name)

            if link == '':
                continue

            youtube_download_opts = {
                'format': 'bestaudio/best', ''
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                'outtmpl': os.path.join(download_dir, music_name) + '.%(ext)s'
            }

            with youtube_dl.YoutubeDL(youtube_download_opts) as ydl:
                ydl.download([link, ])
            download_num += 1
            f_download.write(items[0] + '/' + items[1] + '/' + items[2] + '/' + music_name + '.mp3' + '\n')
            if download_num >= top_n:
                break

        except Exception as e:
            print(str(e))
            f_error.write(items[0] + '/' + items[1] + '\n')       # error 난 음악 출력.

    f.close()
    f_error.close()
    f_download.close()


def download_youtube_mp3_scope(download_dir, file, start_n, num_of_download, dump_dir):
    # ydl_opts['outtmpl'] = r'D:\MyPythonProject\Youtube_DL\download_file\happy_music\%(title)s.%(ext)s'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    f_error = open(os.path.join(dump_dir, 'error.txt'), 'w')
    f_download = open(os.path.join(dump_dir, 'top_download_list.txt'), 'w')
    f = open(file)

    download_num = 0
    idx = 0
    while True:
        line = f.readline()
        print(line)
        if not line:
            break

        if start_n > idx:   # if start_n <= idx: download start!
            idx += 1
            continue

        line = line[:len(line) - 1]  # 맨뒤의 개행문자 삭제
        items = line.split('/')

        music_name = items[1] + ' - ' + items[0]
        print(music_name)

        if download_num % 10 == 0:
            print("Process {} / {} Done".format(download_num, num_of_download))

        try:
            link = search_good_youtube_link(music_name)

            if link == '':
                continue

            youtube_download_opts = {
                'format': 'bestaudio/best', ''
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                'outtmpl': os.path.join(download_dir, music_name) + '.%(ext)s'
            }

            with youtube_dl.YoutubeDL(youtube_download_opts) as ydl:
                ydl.download([link, ])
            download_num += 1
            f_download.write(items[0] + '/' + items[1] + '/' + items[2] + '/' + music_name + '.mp3' + '\n')
            if download_num >= num_of_download:
                break

        except Exception as e:
            print(str(e))
            f_error.write(items[0] + '/' + items[1] + '\n')       # error 난 음악 출력.

    f.close()
    f_error.close()
    f_download.close()


if __name__ == '__main__':
    # download_youtube_mp3_top_n(r'D:\MyPythonProject\music_project\music\happy_top', r'music_data\happy.txt', 200, 'dump\\top_happy')
    # download_youtube_mp3_top_n(r'D:\MyPythonProject\music_project\music\sad_top', r'music_data\sad.txt', 200, 'dump\\top_sad')
    download_youtube_mp3_top_n(r'D:\MyPythonProject\music_project\music\exciting_top', r'music_data\exciting.txt', 200, 'dump\\top_exciting')
    download_youtube_mp3_top_n(r'D:\MyPythonProject\music_project\music\calm_top', r'music_data\calm.txt', 200, 'dump\\top_calm')
    download_youtube_mp3_top_n(r'D:\MyPythonProject\music_project\music\angry_top', r'music_data\angry.txt', 200, 'dump\\top_angry')

    download_youtube_mp3_scope(r'D:\MyPythonProject\music_project\music\happy_test', r'music_data\happy.txt', 230, 20, 'dump\\test_happy')
    download_youtube_mp3_scope(r'D:\MyPythonProject\music_project\music\sad_test', r'music_data\sad.txt', 230, 20, 'dump\\test_sad')
    download_youtube_mp3_scope(r'D:\MyPythonProject\music_project\music\exciting_test', r'music_data\exciting.txt', 230, 20, 'dump\\test_exciting')
    download_youtube_mp3_scope(r'D:\MyPythonProject\music_project\music\calm_test', r'music_data\calm.txt', 230, 20, 'dump\\test_calm')
    download_youtube_mp3_scope(r'D:\MyPythonProject\music_project\music\angry_test', r'music_data\angry.txt', 230, 20, 'dump\\test_angry')


#download_youtube_mp3_2(r'D:\MyPythonProject\music_project\music\happy_song', r'music\happy_song\error.txt')
#download_youtube_mp3_2(r'D:\MyPythonProject\music_project\music\sad_song', r'music_data\sad.txt')
#download_youtube_mp3_2(r'D:\MyPythonProject\music_project\music\exciting_song', r'music_data\exciting.txt')
#download_youtube_mp3_2(r'D:\MyPythonProject\music_project\music\calm_song', r'music_data\calm.txt')
#download_youtube_mp3_2(r'D:\MyPythonProject\music_project\music\angry_song', r'music_data\angry.txt')
