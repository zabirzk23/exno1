# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
```

<img width="1012" height="855" alt="image" src="https://github.com/user-attachments/assets/0f9381ba-ed2f-4023-84e9-d7578879178d" />

```

df.head()
df.tail()
df.head(2)
df.tail(3)

```
<img width="1000" height="187" alt="image" src="https://github.com/user-attachments/assets/d87aeca3-2987-4c44-821e-9367e76171cf" />

```

df.head(2)

```
<img width="934" height="143" alt="image" src="https://github.com/user-attachments/assets/868b3ce8-29e3-4f64-a677-c0fb29628460" />

```

df.head()

```

<img width="985" height="262" alt="image" src="https://github.com/user-attachments/assets/c1e1a28e-bc73-4506-aba1-62d5ec24e343" />

```

df.tail()

```

<img width="1001" height="261" alt="image" src="https://github.com/user-attachments/assets/bd4607b1-1e11-4bfc-9725-ad1d206c9239" />

```

df.isnull()

```

<img width="816" height="855" alt="image" src="https://github.com/user-attachments/assets/76ba2386-288b-43e2-89bb-2c7f4db49b0e" />

```

df.notnull()

```

<img width="790" height="857" alt="image" src="https://github.com/user-attachments/assets/d51e75f0-998a-4176-b3bf-fd74d5a95d06" />

```

df.isnull().sum()

```

<img width="153" height="579" alt="image" src="https://github.com/user-attachments/assets/f772df4e-7ad9-4f01-851f-48dba39f1c49" />

```

df.isnull().any()

```

<img width="182" height="581" alt="image" src="https://github.com/user-attachments/assets/1fb17cce-46ea-47b9-a5a1-382fdf99815f" />

```

df.dropna()

```

<img width="1020" height="573" alt="image" src="https://github.com/user-attachments/assets/5ecdf226-7d22-47cc-b771-d362f8d67b34" />

```

df.dropna(axis=0)

```

<img width="1018" height="554" alt="image" src="https://github.com/user-attachments/assets/828b186e-a534-432f-936a-976ee01ba82b" />

```

df.dropna(axis=1)

```

<img width="282" height="853" alt="image" src="https://github.com/user-attachments/assets/c4f590bc-f1ef-41f3-8b7a-a0be5a3ea46f" />

```

df.fillna(5)

```

<img width="1016" height="859" alt="image" src="https://github.com/user-attachments/assets/e43a6a2b-9f07-4f89-9703-e27a4f3b90f8" />

```

df.fillna(method='ffill')

```

<img width="1011" height="857" alt="image" src="https://github.com/user-attachments/assets/199bce10-e65e-4d7e-a890-15aae9e992d4" />

```

df.fillna(method='bfill')

```

<img width="1013" height="856" alt="image" src="https://github.com/user-attachments/assets/2f42e4a0-2a8a-42f9-a3de-2016ccfdf1d6" />

```

df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':96,'M3':87,'M4':90,'TOTAL':487,'AVG':90})

```

<img width="1018" height="860" alt="image" src="https://github.com/user-attachments/assets/ff1573ae-b492-4e2e-ada5-c4a929ca9834" />

```

import seaborn as sns
ir=pd.read_csv("iris.csv")
ir

```

<img width="672" height="533" alt="image" src="https://github.com/user-attachments/assets/26ba9566-64e3-4c48-94be-ada98df683cf" />

```

ir.describe()

```

<img width="598" height="383" alt="image" src="https://github.com/user-attachments/assets/c343562c-7464-42c2-be1d-5d7fc6dc709b" />

```

import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)

```

<img width="679" height="583" alt="image" src="https://github.com/user-attachments/assets/db220b6c-586b-4861-bbd1-b213cfb0a13a" />

```

q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iq=q3-q1
print(iq)

```

<img width="675" height="45" alt="image" src="https://github.com/user-attachments/assets/f905be5e-7f8f-43fd-ad69-31aa3192c887" />

```

rid=ir[((ir.sepal_width<(q1-1.5*iq))| (ir.sepal_width>(q3+1.5*iq)))]
rid['sepal_width']

```

<img width="181" height="252" alt="image" src="https://github.com/user-attachments/assets/d3c6d872-b1fc-42af-9332-1e004434259f" />

```

delid=ir[~((ir.sepal_width<(q1-1.5*iq))| (ir.sepal_width>(q3+1.5*iq)))]
delid

```

<img width="661" height="519" alt="image" src="https://github.com/user-attachments/assets/349bea3c-1e37-4afe-a534-dc0e6eeba369" />

```

sns.boxplot(x='sepal_width',data=delid)

```

<img width="650" height="568" alt="image" src="https://github.com/user-attachments/assets/6ad79408-0405-4901-8f2b-e03acc346978" />

```

import numpy as np
import scipy.stats as stats
daf=pd.read_csv("heights.csv")
daf

```

<img width="220" height="607" alt="image" src="https://github.com/user-attachments/assets/0f3c4c54-51b6-42cc-af83-f551a437a994" />

```

z=np.abs(stats.zscore(daf['height']))
z

```

<img width="673" height="86" alt="image" src="https://github.com/user-attachments/assets/b212bfcf-e6e6-4c95-903f-dfc9094274a2" />

```

df1=daf[z<3]
df1

```

<img width="209" height="572" alt="image" src="https://github.com/user-attachments/assets/b016b997-2b84-4d0f-9812-e059a3ed21e3" />


# Result

Hence the data was cleaned , outliers were detected and removed.
