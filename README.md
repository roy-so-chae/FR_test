# Video Object Segmentation
## OSVOS(One-Shot Video Object Segmentation) [[Paper]](https://arxiv.org/pdf/1611.05198.pdf)
- 각 frame을 독립적으로 연산하며 Time sequence상에서 continuous하게 correlation을 활용하지 않는다.

Frame-based processing introduces temporal inconsistencies 

![캡처1](https://user-images.githubusercontent.com/74402562/117539295-fc2cef00-b044-11eb-85d3-437cbdd2a27f.PNG)

But results are still very convincing

![캡처2](https://user-images.githubusercontent.com/74402562/117539299-fe8f4900-b044-11eb-9cac-7df6a49b0101.PNG)

------------------------------------------
## MaskTrack(Learning Video Object Segmentation from Static Images) [[Paper]](https://arxiv.org/pdf/1612.02646.pdf)
- Input fame과 이전 mask estimate를 concatenate해서 네트워크를 통과시키면 이전 mask를 refine하여 input frame에 맞는 mask를 생성한다.
- 이전 영상과 다음 frame의 연관성을 활용하여 mask를 생성한다.

The ConvNet is trained to refine the previous mask to the current frame

![캡처3](https://user-images.githubusercontent.com/74402562/117539300-ffc07600-b044-11eb-93e6-e9163ff956a7.PNG)

---------------------------------------
## RGMP(Fast Video Object Segmentation by Reference-Guided Mask Propagation) [[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Oh_Fast_Video_Object_CVPR_2018_paper.pdf)
- 이전 frame과 mask뿐만 아니라 Reference frame과 mask(첫번째 frame의 이미지와 mask정보)를 함께 넘겨준다.

![캡처4](https://user-images.githubusercontent.com/74402562/117539303-018a3980-b045-11eb-850d-41412f1e72bf.PNG)

----------------------------------------
## RVOS(End-to-End Recurrent Network for Video Object Segmentation) [[Paper]](https://arxiv.org/pdf/1903.05612.pdf)
- Spartial recurrence와 Temporal recurrence를 모두 고려한다.
- Frame sequence뿐만 아니라 Object sequence도 고려한다. 예) 사람-말-개 순서로 마스크를 생성
![캡처5](https://user-images.githubusercontent.com/74402562/117539306-02bb6680-b045-11eb-9d2b-74cb360f0484.PNG)

![캡처6](https://user-images.githubusercontent.com/74402562/117539309-03ec9380-b045-11eb-8ef9-03e241ff2f4a.PNG)

![캡처7](https://user-images.githubusercontent.com/74402562/117539312-05b65700-b045-11eb-84b8-a529e8120fdf.PNG)

-----------------------------------------
## STMN(Video Object Segmentation using Space-Time Memory Networks) [[Paper]](https://arxiv.org/pdf/1904.00607.pdf)
- 여러 개의 intermediate frame을 사용하는 방법을 제안한다.

![캡처8](https://user-images.githubusercontent.com/74402562/117539313-06e78400-b045-11eb-9898-9bc5e9995b42.PNG)

- Memory구조를 활용하여 이전 Frame들의 정보를 저장하고, 현재 Frame에 도움되는 부분을 읽어서 사용한다.
- 아래 그림의 왼쪽 부분처럼 이전 RGB frame과 foreground mask가 memory에 저장되고, 오른쪽 부분에서는 현재 주어진 frame을 기반으로 memory에서 정보를 가져온 후 mask를 예측한다.
![캡처9](https://user-images.githubusercontent.com/74402562/117539314-08b14780-b045-11eb-9862-ffe608732ac2.PNG)

- Query key와 memory key의 similarity를 dot product로 계산한 후 softmax값을 memory value과 weighted sum을 한다. 이렇게 read된 memory는 query value와 concat되어 최종 read output이 된다.

![캡처10](https://user-images.githubusercontent.com/74402562/117539319-0a7b0b00-b045-11eb-9b6e-a579f5db0ceb.PNG)
