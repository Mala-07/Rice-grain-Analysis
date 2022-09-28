```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
def get_classificaton(ratio):
    ratio =round(ratio,1)
    toret=""
    if(ratio>=3):
        toret="Slender"
    elif(ratio>=2.1 and ratio<3):
        toret="Medium"
    elif(ratio>=1.1 and ratio<2.1):
        toret="Bold"
    elif(ratio<=1):
        toret="Round"
    toret="("+toret+")"
    return toret
#rnjn
print("Starting")
```

    Starting
    


```python
I = cv2.imread('image_1.jpg')
gray= cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5), 0)
plt.imshow(I)
```




    <matplotlib.image.AxesImage at 0x26701190310>




    
![png](output_2_1.png)
    



```python
bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,1)
plt.imshow(bw)
plt.savefig('pics/THRESHOLD_1.png')
```


    
![png](output_3_0.png)
    



```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
plt.imshow(kernel)
```




    <matplotlib.image.AxesImage at 0x2670131b610>




    
![png](output_4_1.png)
    



```python
bw_1 = cv2.dilate(cv2.erode(bw, kernel), kernel)
plt.imshow(bw_1)
plt.savefig('pics/DILATED_1.png')
```


    
![png](output_5_0.png)
    



```python
(contours,heirarchy) = cv2.findContours(bw_1.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb  = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, contours,-1,(0,225,0),3)
plt.imshow(rgb)
plt.savefig('pics/RGBimg_1.png')
```


    
![png](output_6_0.png)
    



```python
print('no. of grains:',(len(contours)))
```

    no. of grains: 539
    


```python
import pandas as pd
```


```python
total_ar=0
df = pd.DataFrame(columns=["ratio","classification"])
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if(aspect_ratio<1):
        aspect_ratio=1/aspect_ratio
        df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    print(round(aspect_ratio,2),get_classificaton(aspect_ratio))
    total_ar+=aspect_ratio
avg_ar=total_ar/len(contours)
print("Average Aspect Ratio=",round(avg_ar,2),get_classificaton(avg_ar))

```

    2.77 (Medium)
    2.36 (Medium)
    1.73 (Bold)
    2.67 (Medium)
    3.88 (Slender)
    3.29 (Slender)
    1.4 (Bold)
    1.0 (Round)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.04 (Round)
    1.05 (Bold)
    1.06 (Bold)
    1.38 (Bold)
    1.02 (Round)
    1.45 (Bold)
    1.1 (Bold)
    1.23 (Bold)
    2.11 (Medium)
    1.92 (Bold)
    2.29 (Medium)
    1.07 (Bold)
    1.14 (Bold)
    1.5 (Bold)
    1.0 (Round)
    1.23 (Bold)
    1.03 (Round)
    1.03 (Round)
    1.24 (Bold)
    1.24 (Bold)
    1.41 (Bold)
    1.65

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.19 (Bold)
    1.9 (Bold)
    1.03 (Round)
    2.02 (Bold)
    1.12 (Bold)
    1.16 (Bold)
    1.3 (Bold)
    1.18 (Bold)
    1.32 (Bold)
    1.0 (Round)
    1.22 (Bold)
    1.04 (Round)
    1.61 (Bold)
    1.29 (Bold)
    1.51 (Bold)
    1.65 (Bold)
    1.12 (Bold)
    1.14 (Bold)
    1.38 (Bold)
    1.35 (Bold)
    1.14 (Bold)
    1.34 (Bold)
    1.15 (Bold)
    1.61 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    2.5 (Medium)
    1.22 (Bold)
    1.64 (Bold)
    1.3 (Bold)
    1.49 (Bold)
    2.19 (Medium)
    1.29 (Bold)
    2.39 (Medium)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.02 (Round)
    1.08 (Bold)
    1.28 (Bold)
    2.08 (Medium)
    1.49 (Bold)
    1.19 (Bold)
    1.14 (Bold)
    1.36 (Bold)
    1.39 (Bold)
    1.04 (Round)
    2.17

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Medium)
    1.94 (Bold)
    1.15 (Bold)
    2.16 (Medium)
    1.49 (Bold)
    1.29 (Bold)
    1.19 (Bold)
    1.51 (Bold)
    1.88 (Bold)
    2.19 (Medium)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.12 (Bold)
    1.1 (Bold)
    1.78 (Bold)
    1.45 (Bold)
    2.47 (Medium)
    1.76 (Bold)
    1.36 (Bold)
    1.14 (Bold)
    1.79 (Bold)
    1.04 (Round)
    1.12 (Bold)
    1.08 (Bold)
    1.1 (Bold)
    1.91 (Bold)
    1.08 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.23 (Bold)
    2.26 (Medium)
    1.3 (Bold)
    2.12 (Medium)
    1.4 (Bold)
    1.44 (Bold)
    2.19 (Medium)
    1.04 (Round)
    1.46 (Bold)
    1.75 (Bold)
    1.57 (Bold)
    1.26 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.24 (Bold)
    1.94 (Bold)
    1.05 (Round)
    1.05 (Bold)
    1.64 (Bold)
    1.0 (Round)
    1.12

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.03 (Round)
    1.18 (Bold)
    1.02 (Round)
    2.58 (Medium)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    2.11 (Medium)
    1.31 (Bold)
    1.13

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.77 (Bold)
    2.24 (Medium)
    1.61 (Bold)
    1.58 (Bold)
    1.17

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.23 (Bold)
    1.46 (Bold)
    1.34 (Bold)
    1.0 (Round)
    1.04 (Round)
    1.48 (Bold)
    1.33 (Bold)
    1.04 (Round)
    1.6 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.77 (Bold)
    1.09 (Bold)
    1.26 (Bold)
    1.68 (Bold)
    2.24 (Medium)
    1.68

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.55 (Bold)
    2.3 (Medium)
    1.12 (Bold)
    1.84 (Bold)
    1.52 (Bold)
    1.65 (Bold)
    1.55 (Bold)
    2.12 (Medium)
    2.33 (Medium)
    3.1 (Slender)
    1.45 (Bold)
    1.24 (Bold)
    2.06 (Medium)
    1.3 (Bold)
    1.1 (Bold)
    1.24 (Bold)
    2.0 (Bold)
    1.07

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.85 (Bold)
    1.07 (Bold)
    1.16 (Bold)
    1.37 (Bold)
    1.11 (Bold)
    1.19 (Bold)
    2.13 (Medium)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.09 (Bold)
    1.09 (Bold)
    1.18 (Bold)
    1.11 (Bold)
    1.51 (Bold)
    1.2 (Bold)
    1.94 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.35 (Bold)
    1.28 (Bold)
    2.16 (Medium)
    1.19 (Bold)
    1.04 (Round)
    1.26 (Bold)
    1.13 (Bold)
    1.18 (Bold)
    1.47 (Bold)
    1.03 (Round)
    1.43 (Bold)
    1.26 (Bold)
    1.39 (Bold)
    1.04

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Round)
    1.72 (Bold)
    1.69 (Bold)
    1.15 (Bold)
    1.58 (Bold)
    1.04 (Round)
    1.53 (Bold)
    1.83 (Bold)
    1.25 (Bold)
    1.0 (Round)
    1.76 (Bold)
    1.26 (Bold)
    1.8 (Bold)
    1.2 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.08 (Bold)
    1.21 (Bold)
    2.58 (Medium)
    1.69 (Bold)
    1.13 (Bold)
    1.3 (Bold)
    1.44 (Bold)
    1.21 (Bold)
    1.18 (Bold)
    1.44 (Bold)
    1.97 (Bold)
    1.58 (Bold)
    1.56 (Bold)
    1.16 (Bold)
    1.31 (Bold)
    2.07 (Medium)
    1.93 (Bold)
    1.15

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.85 (Bold)
    1.22 (Bold)
    1.41 (Bold)
    1.15 (Bold)
    2.39 (Medium)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.46 (Bold)
    1.04 (Round)
    1.06 (Bold)
    1.62 (Bold)
    1.68 (Bold)
    1.0 (Round)
    1.5 (Bold)
    1.44 (Bold)
    1.07 (Bold)
    1.23 (Bold)
    1.94 (Bold)
    1.18 (Bold)
    1.19 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.41 (Bold)
    1.44 (Bold)
    1.05 (Round)
    1.06 (Bold)
    1.56 (Bold)
    1.85 (Bold)
    1.07 (Bold)
    1.56 (Bold)
    1.21

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.17 (Bold)
    1.1 (Bold)
    1.76 (Bold)
    1.95 (Bold)
    1.28 (Bold)
    1.55 (Bold)
    1.88 (Bold)
    1.66 (Bold)
    1.16 (Bold)
    1.17 (Bold)
    1.61 (Bold)
    1.26 (Bold)
    1.05 (Bold)
    1.11 (Bold)
    1.62 (Bold)
    1.0 (Round)
    1.52

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.18 (Bold)
    1.8 (Bold)
    1.25 (Bold)
    1.89 (Bold)
    1.59 (Bold)
    1.12 (Bold)
    1.37 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.06 (Bold)
    1.91 (Bold)
    1.77 (Bold)
    1.0 (Round)
    1.81 (Bold)
    1.8 (Bold)
    1.04 (Round)
    1.25 (Bold)
    1.11 (Bold)
    2.36 (Medium)
    1.81 (Bold)
    1.92 (Bold)
    1.04 (Round)
    1.02 (Round)
    1.8 (Bold)
    1.38 (Bold)
    1.3 (Bold)
    1.03 (Round)
    2.13 (Medium)
    1.08 (Bold)
    1.07 (Bold)
    2.2

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Medium)
    1.31 (Bold)
    2.22 (Medium)
    1.18 (Bold)
    1.08 (Bold)

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    
    1.75 (Bold)
    1.48 (Bold)
    1.32 (Bold)
    1.45 (Bold)
    1.53 (Bold)
    1.07 (Bold)
    1.08 (Bold)
    1.07 (Bold)
    1.97 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.38 (Bold)
    1.08 (Bold)
    1.44 (Bold)
    1.76 (Bold)
    1.94 (Bold)
    1.22 (Bold)
    1.43 (Bold)
    2.41 (Medium)
    1.08

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.15 (Bold)
    2.23 (Medium)
    2.19 (Medium)
    2.0 (Bold)
    2.39 (Medium)
    1.97 (Bold)
    2.56 (Medium)
    1.06 (Bold)
    2.78 (Medium)
    1.17 (Bold)
    2.05 (Bold)
    1.33 (Bold)
    1.7 (Bold)
    1.35 (Bold)
    1.71 (Bold)
    2.64 (Medium)
    2.21 (Medium)
    1.33

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.03 (Round)
    1.48 (Bold)
    1.16 (Bold)
    1.07 (Bold)
    1.97 (Bold)
    1.83 (Bold)
    2.09 (Medium)
    1.08 (Bold)
    1.2 (Bold)
    1.47 (Bold)
    1.76 (Bold)
    1.11 (Bold)
    1.35

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.12 (Bold)
    1.3 (Bold)
    1.29 (Bold)
    2.81 (Medium)
    1.17 (Bold)
    1.04 (Round)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.79 (Bold)
    1.53 (Bold)
    1.44 (Bold)
    2.02 (Bold)
    2.03 (Bold)
    1.02 (Round)
    1.04 (Round)
    1.05 (Bold)
    1.35 (Bold)
    2.17 (Medium)
    1.33 (Bold)
    2.29 (Medium)
    1.32 (Bold)
    1.38 (Bold)
    1.27 (Bold)
    2.16 (Medium)
    1.39 (Bold)
    2.26 (Medium)
    1.77 (Bold)
    2.0 (Bold)
    1.38 (Bold)
    1.0 (Round)
    1.11 (Bold)
    1.35 (Bold)
    1.24 (Bold)
    1.26 (Bold)
    1.18 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.3 (Bold)
    1.17 (Bold)
    1.65 (Bold)
    1.07 (Bold)
    1.16 (Bold)
    1.11 (Bold)
    1.55 (Bold)
    2.31 (Medium)
    2.16

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Medium)
    1.24 (Bold)
    1.25 (Bold)
    1.97 (Bold)
    1.84 (Bold)
    1.18 (Bold)
    1.83 (Bold)
    1.53 (Bold)
    1.12 (Bold)
    1.82 (Bold)
    1.93 (Bold)
    1.16 (Bold)
    2.35 (Medium)
    1.42 (Bold)
    1.49 (Bold)
    1.59 (Bold)
    1.02 (Round)
    1.14 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    2.11 (Medium)
    1.06 (Bold)
    1.36 (Bold)
    2.0 (Bold)
    1.59 (Bold)
    2.13 (Medium)
    1.73 (Bold)
    1.02 (Round)
    1.77 (Bold)
    1.33 (Bold)
    2.16 (Medium)
    1.35

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.03 (Round)
    1.72 (Bold)
    1.38 (Bold)
    1.78 (Bold)
    1.58 (Bold)
    1.16 (Bold)
    1.3 (Bold)
    1.4 (Bold)
    1.81 (Bold)
    1.31 (Bold)
    1.07 (Bold)
    1.94 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.38 (Bold)
    1.41 (Bold)
    1.25 (Bold)
    1.5 (Bold)
    1.01 (Round)
    1.4 (Bold)
    1.16 (Bold)
    1.41 (Bold)
    1.1

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.53 (Bold)
    1.29 (Bold)
    2.03 (Bold)
    1.12 (Bold)
    1.17 (Bold)
    1.97 (Bold)
    3.19 (Slender)
    1.81

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.16 (Bold)
    1.09 (Bold)
    1.39 (Bold)
    1.11 (Bold)
    1.4 (Bold)
    1.07 (Bold)
    1.49 (Bold)
    1.69 (Bold)
    1.07 (Bold)
    1.61 (Bold)
    1.09 (Bold)
    1.32 (Bold)
    1.36 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.07 (Bold)
    1.22 (Bold)
    1.03 (Round)
    1.87 (Bold)
    1.04 (Round)
    2.1 (Medium)
    2.9 (Medium)
    1.92 (Bold)
    1.4 (Bold)
    1.85 (Bold)
    2.23 (Medium)
    1.19 (Bold)
    1.91 (Bold)
    1.43

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.22 (Bold)
    2.56 (Medium)
    1.03 (Round)
    3.34 (Slender)
    2.03 (Bold)
    1.85 (Bold)
    1.89 (Bold)
    1.1 (Bold)
    1.23 (Bold)
    1.3 (Bold)
    1.63 (Bold)
    1.42 (Bold)
    1.33 (Bold)
    1.79 (Bold)
    1.1 (Bold)
    1.18 (Bold)
    1.27 (Bold)
    1.94

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    2.91 (Medium)
    1.33 (Bold)
    1.29 (Bold)
    1.1 (Bold)
    2.32 (Medium)
    1.18 (Bold)
    1.5 (Bold)
    1.92 (Bold)
    1.21 (Bold)
    1.47 (Bold)
    1.96 (Bold)
    1.94 (Bold)
    1.48 (Bold)
    1.42

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

     (Bold)
    1.1 (Bold)
    1.31 (Bold)
    1.59 (Bold)
    1.56 (Bold)
    1.43 (Bold)
    1.1 (Bold)
    1.21 (Bold)
    1.24 (Bold)
    1.74 (Bold)
    2.09 (Medium)
    1.25 (Bold)
    2.07 (Medium)
    1.97 (Bold)
    1.12 (Bold)
    1.3 (Bold)
    1.23 (Bold)
    1.11 (Bold)
    1.08 (Bold)
    1.53 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    

    1.05 (Bold)
    1.76 (Bold)
    1.32 (Bold)
    1.27 (Bold)
    1.69 (Bold)
    1.02 (Round)
    2.78 (Medium)
    1.92 (Bold)
    1.32 (Bold)
    1.08 (Bold)
    1.23 (Bold)
    1.68 (Bold)
    1.16 (Bold)
    1.08 (Bold)
    2.0 (Bold)
    Average Aspect Ratio= 1.5 (Bold)
    

    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    C:\Users\SONY\AppData\Local\Temp\ipykernel_9588\745013339.py:8: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df = df.append({"ratio":round(aspect_ratio,2),"classification":get_classificaton(aspect_ratio)},ignore_index=True)
    


```python
df = df.sort_values(by='ratio')
```


```python
data = pd.DataFrame(df['classification'].value_counts())
data= data.T
data.rename(columns={'(Bold)':'Bold','(Medium)':'Medium','(Round)':'Round','(Slender)':'Slender'}, inplace=True)
data.rename(index={'classification':'count'},inplace = True)
data
#data.rename(columns={'index':'classification','classification':'count'}, inplace=True)
#data = data.T
#data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bold</th>
      <th>Medium</th>
      <th>Round</th>
      <th>Slender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>223</td>
      <td>28</td>
      <td>23</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('classification_1')
```


```python
data.to_csv('grain_count_1')
```


```python

```
