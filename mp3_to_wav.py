
import pydub
from pydub import AudioSegment

duration = 60 * 3 * 1000



sound = AudioSegment.from_mp3("D:\\MyPythonProject\music_project\\트와이스.mp3")     # for happy_song.mp3
sound.export(r"D:\MyPythonProject\music_project\트와이스.wav", format="wav")

