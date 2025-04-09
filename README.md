# VoiceStylingModel
![image](https://github.com/pzxcvasd/VoiceStylingModel/assets/139040422/75fc5d59-b835-4ec3-80db-2b5fd5cfc59d)

최종 통합 코드: talking yu.ipnyb
</br>
해당 주제 투고한 논문: 
[입력 텍스트로부터 음성 스타일을 반영할 수 있는 가상 인간 연구.pdf](https://github.com/user-attachments/files/19654823/default.pdf)
</br> 
보고서:[[3]입력텍스트를 읽어주는 voice styling 기반 딥페이크 모델 보고서.docx](https://github.com/user-attachments/files/19658454/3.voice.styling.docx)



## DeepFake
<img src="https://github.com/pzxcvasd/VoiceStylingModel/assets/139040422/0bf4583c-e800-4e49-a622-67e78f1d48fe" width="200" /> </br>
### 1) DeepFaceLab 사용
Github : https://github.com/iperov/DeepFaceLab </br>
정확성이 높은 **SAEHD** 모델을 사용해 유재석 얼굴과 뉴스앵커 얼굴을 추출 후 학습을 진행한다. </br>
<img src="https://github.com/pzxcvasd/project_backup/assets/99024754/05dfb1a0-b330-46f9-9d86-3dbb715eccb1" width ="250" /> </br>
이후 나온 결과물(dfm) 을 바탕으로 DeepFaceLive 에서 face-swap 을 진행한다.

### 2) DeepFaceLive 적용 영상
각종 후처리 작업을 끝낸 최종 딥페이크 모델 결과물이다. </br>
<img src="https://github.com/pzxcvasd/project_backup/assets/99024754/3d468528-6760-49da-a3ef-2c3c52697fb3" width="500" />


## TTS

결과물 sample은 파일들 중 .wav 파일 참조. 

참조: https://github.com/serp-ai/bark-with-voice-clone 와 https://github.com/JonathanFly/bark

작업 환경: colab pro+ / google drive more than 200GB

코드: bark yoojs TTS.ipynb

코드 구성:
![image](https://github.com/pzxcvasd/VoiceStylingModel/assets/139040422/51dd30af-1338-403d-a84b-03b78b955f5c)


## LipSync

![image](https://github.com/pzxcvasd/VoiceStylingModel/assets/139040422/23b38eee-cd7c-4ca6-822b-6c6a376b5e7f)


https://github.com/pzxcvasd/VoiceStylingModel/assets/144596857/ff8bc77e-a1bb-4d68-af43-866ba3722672



https://github.com/pzxcvasd/VoiceStylingModel/assets/144596857/3206b654-dba9-4a53-9439-3b912b4e1ebf


