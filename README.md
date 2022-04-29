# [AI Tech 3기 Level2 P Stage] 글자 검출 대회

<img width="1100" alt="image" src="https://user-images.githubusercontent.com/57162812/165903899-92b14609-f1fc-40f5-b100-1af85be6743e.png">

## 팀원 소개

|김규리_T3016|박정현_T3094|석진혁_T3109|손정균_T3111|이현진_T3174|임종현_T3182|
|:-:|:-:|:-:|:-:|:-:|:-:|
|||||

## Overview

OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있는데  본 대회는 글자 검출 (text detection)만을 해결하게 됩니다.

데이터를 구성하고 활용하는 방법에 집중하는 것을 장려하는 취지에서, 제공되는 베이스 코드 중 모델과 관련한 부분을 변경하는 것이 금지되어 있습니다.  데이터 수집과 preprocessing, data augmentation 그리고 optimizer, learning scheduler 등 최적화 방식을 변경할 수 있습니다.

- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format

### 평가방법

- DetEval
    
    이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법 중 하나 입니다
    
    1. **모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산해냅니다.**
    2. **모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정합니다.**
    3. **모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정됩니다.**
        
        ![image](https://user-images.githubusercontent.com/57162812/165904753-87823a80-8042-4c4e-8567-327b7b057914.png)
        

### **Final Score 🏅**

- Public : f1 0.6897 → Private f1 : **0.6751**
- Public : 11위/19팀 → Private : 9**위/19팀**

![image](https://user-images.githubusercontent.com/57162812/165904420-740c131f-eb74-4dac-aee6-150dca012d47.png)

## **Archive contents**

```python
template
├──code
│  ├──augmentation.py
│  ├──convert_mlt.py
│  ├──dataset.py
│  ├──deteval.py
│  ├──east_dataset.py
│  ├──inference.py
│  ├──loss.py
│  ├──model.py
│  └──train.py
└──input
   └──ICDAR2017_Korean
		  └──data
			  	├──images
		      └──ufo
			        ├──train.json
							└──val.json
```

## Dataset

- [ ]  ICDAR MLT17 Korean : 536 images ⊆ ICDAR MLT17 : 7,200 images
- [ ]  ICDAR MLT19 : 10,000 images
- [ ]  ICAR ArT : 5,603 images
    
    <img src="https://user-images.githubusercontent.com/57162812/165904462-381ab17d-7f0c-49a1-a400-f25c1740a40a.png" width="60%">
    

## Experiment

- [wrapup report](https://www.notion.so/CV-Wrap-Up-Report-1d8fe4b84edb4c8489f4c93bfba8bb19)

## Results

|  | dataset | 데이터 수 | LB score (public→private) | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 01 | ICDAR17_Korean | 536 | 0.4469 → 0.4732 | 0.3580 → 0.3803 | 0.5944 → 0.6264 |
| 02 | Camper (폴리곤 수정 전) | 1288 | 0.4543 → 0.5282 | 0.3627 → 0.4349 | 0.6077 → 0.6727 |
| 03 | Camper (폴리곤 수정 후) | 1288 | 0.4644 → 0.5298 | 0.3491 → 0.4294 | 0.6936 → 0.6913 |
| 04 | ICDAR17_Korean + Camper | 1824 | 0.4447 → 0.5155 | 0.3471 → 0.4129 | 0.6183 → 0.6858 |
| 05 | ICDAR17(859) | 859 | 0.5435 → 0.5704 | 0.4510 → 0.4713 | 0.6837 → 0.7222 |
| 06 | ICDAR17_MLT | 7200 | 0.6749 → 0.6751 | 0.5877 → 0.5887 | 0.7927 → 0.7912 |
| 07 | ICDAR19+ArT | 약 15000 | 0.6344 → 0.6404 | 0.5489 → 0.5607 | 0.7514 → 0.7465 |

## Requirements

```jsx
pip install -r requirements.txt
```

## UFO Format으로 변환

```jsx
python convert_mlt.py
```

SRC_DATASET_DIR = {변환 전 data 경로}

DST_DATASET_DIR = {변환 된 data 경로}

## UFO Format ****

```python
File Name
    ├── img_h
    ├── img_w
    └── words
        ├── points
        ├── transcription
        ├── language
        ├── illegibillity
        ├── orientation
        └── word_tags
```

## Train.py

```jsx
python train.py --data_dir {train data path} --val_data_dir {val data path} --name {wandb run name} --exp_name {model name
```


