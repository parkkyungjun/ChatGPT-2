# 소개
- gpt-2를 pre-train, step1, 2, 3에 대해서 학습하는 코드입니다.
- pre-train은 10.795841 M parameters의 크기의 모델로 셰익스피어 데이터셋을 학습했습니다. (decoder block 6개)
- RLHF를 진행한 step1, 2, 3은 gpt-2-small의 pre-train모델의 가중치를 사용했습니다.
- step1의 SFL을 위한 데이터셋은 한국어 QA의 질의, 응답은 chatgpt를 사용하였습니다.
```
ex)
{
    "prompt": "술 먹고 싶어",
    "completion": "'술은 알코올이 함유된 음료수이며, 건강에 나쁜 영향을 미칠 수 있습니다. 따라서 술을 마시기 전에 본인의 건강 상태와 함께 적당한 양을 마시는 것이 중요합니다. 술을 마시기 전에 반드시 20세 이상인지 확인하시고, 규정된 양 이내에서 적절하게 즐기시길 바랍니다. 또한 술을 마시고 운전하는 것은 절대 금지하시기 바랍니다.",
    "tokens": 189
}
```
- step2의 reward를 위한 rank는 chatgpt가 생성한 응답이 1등 2등은 gpt-davinci, 3등은 gpt-ada으로 사용하였습니다. <br/><br/><br/><br/>

![](https://openaicom.imgix.net/cf717bdb-0c8c-428a-b82b-3c3add87a600/ChatGPT_Diagram.svg?fm=auto&auto=compress,format&fit=min&w=1919&h=1138)
## Pre-train

10.795841 M parameters<br/>
### Result
CAPULET:<br/>
A fruit-cold foolish sound, only foe's change not<br/>
Romeo behind: your honour is not my time,<br/>
There seems a testimonion; therefore I'll be<br/>
More hope to you for curses; and tell your own parture<br/>
By him doing that it were as mine.<br/>
<br/>
CAMILLO:<br/>
My lord, sir,<br/>
You must not take it.<br/>
<br/>
POLIXENES:<br/>
Now yet she hath, my father<br/>
I would be so too: but I would indeed, I'll buy<br/>
The horses being o' the loss, nor there;<br/>
Or, now I'ld we were never speak'd<br/>
We all reason the breeders to not: come, you have fear<br/>
<br/>
어느정도 영어 문법과 단어를 성공적으로 학습하였습니다.

## step1
## step2
## step3
