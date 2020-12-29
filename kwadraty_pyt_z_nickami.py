#!/usr/bin/python3
import requests
import pandas as pd
import os
import time
from datetime import datetime
from requests.exceptions import HTTPError
import json
import numpy as np
from numpy import unravel_index


def dp(martix):
    m, n = len(martix), len(martix[0])
    cache = np.array(([[0 for x in range(n)] for x in range(m)]),dtype=np.uint8)
    for x in range(m):
        cache[x][0] = martix[x][0]
    for x in range(n):
        cache[0][x] = martix[0][x]
    #largestSubArray = 0
    for i in range(1, m):
        for j in range(1, n):
            if martix[i][j - 1] == 1 and martix[i - 1][j] == 1 and martix[i - 1][j - 1] == 1:
                cache[i][j] = min(cache[i - 1][j], cache[i][j - 1], cache[i - 1][j - 1])+1
            else:
                cache[i][j] = martix[i][j]
            #largestSubArray = max(largestSubArray, cache[i][j])
    #for x in range(m):
    #    print(cache[x])
 
    return np.array(cache,dtype=np.uint8)
    #return largestSubArray

def maximalSquare(matrix):
    nrows = len(matrix)
    ncols = len(matrix[0])
    max_square_len = 0
    dp = [[0] * (ncols + 1) for i in range(nrows + 1)]

    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            if (matrix[i - 1][j - 1] == True):
                dp[i][j] = min(dp[i][j - 1], dp[i - 1]
                               [j], dp[i - 1][j - 1]) + 1
                max_square_len = max(max_square_len, dp[i][j])
    #print(np.array(dp))
    #print(np.array(dp).argmax(axis=None))
    #print(np.unravel_index(np.array(dp).argmax(axis=None), (np.array(dp).shape)))
    #return max_square_len ** 2
    return np.array(dp,dtype=np.uint8)

def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values
  
def removeDuplicates(lst): 
      
    return list(set([i for i in lst])) 



lista_linkow_po_roz = []

startTime = datetime.now()

# kontener na wczytane kwadraty - sparwdz czy plik istnieje jesli nie to kontener to pusty dataframe jesli tak zaladuj plik do kontenera
dane_kwadraty_istnieja = os.path.isfile('./linki.json')
if dane_kwadraty_istnieja == True:
    uid_data = pd.read_json('linki.json')
else:
    print("Ni ma pliku wejściowego!")


try:

    tuple_global = []
    Lista_max_square = pd.DataFrame()
    


    print("Rozwijanie linków:")

    for item in range(len(uid_data)) :

        iksy_uzyt = []
        ygreki_uzyt = []
        id_uzyt = pd.DataFrame()
        

        response = requests.get(uid_data.loc[item, "link"] +'/api/activities?page=1')
        response.raise_for_status()
        # access JSOn content
        jsonResponse = response.json()
        print('Przetwarzanie usera: '+ uid_data.loc[item, "nick"])
        print("Strona 1: Znalezionych treningow: ",len(jsonResponse["activities"]))
        #nowy = []
        #for element in jsonResponse["activities"]:
            #del element["activities"]["idb_metric"]
            #print(element["type"])
        #    if element["type"] != "Ride":
        #        nowy.append(element)

        #iksy_uzyt += json_extract(nowy, 'x')
        #ygreki_uzyt += json_extract(nowy, 'y')
        
        iksy_uzyt += json_extract(jsonResponse, 'x')
        ygreki_uzyt += json_extract(jsonResponse, 'y')
        
        counter = 1
        time.sleep(3)
        while len(jsonResponse["activities"]) == 500:
            counter += 1
            response = requests.get(uid_data.loc[item, "link"] +'/api/activities?page='+ str(counter))
            response.raise_for_status()
            # access JSOn content
            jsonResponse = response.json()
            print("Strona "+str(counter)+" Znalezionych treningow: ",len(jsonResponse["activities"]))
            #nowy = []
            #for element in jsonResponse["activities"]:
                #del element["activities"]["idb_metric"]
                #print(element["type"])
            #    if element["type"] != "Ride":
            #        nowy.append(element)
            #left top
            #7700 4700
            #bott rig
            #10000 6500

            iksy_uzyt += json_extract(jsonResponse, 'x')
            ygreki_uzyt += json_extract(jsonResponse, 'y')
            #time.sleep(5)
        lista_nickow = [uid_data.loc[item, "uid"]]*len(iksy_uzyt)
        polaczone_uzyt = [tuple(x) for x in zip(iksy_uzyt, ygreki_uzyt, lista_nickow)]
        przefiltrowane_uzyt = [x for x in polaczone_uzyt if (x[0]>7700 and x[0]<10000 and x[1]>4700 and x[1]<6500)]
        unikalne_uzyt = removeDuplicates(przefiltrowane_uzyt)
        
        #przefiltrowane_uzyt = filter((lambda x: x[0] > 7700,lambda x: x[0] < 10000,lambda x: x[1] > 4700,lambda x: x[1] < 6500),unikalne_uzyt)

        
        print("Wyliczanie max_square")
        #wyliczanie max_sqr uzytkowanika
        df = pd.DataFrame(unikalne_uzyt,columns=['x','y','uid'])
        df = df.drop(columns=['uid'])
        
        col_one_arr = df['x'].to_numpy(dtype=np.int16)
        row_one_arr = df['y'].to_numpy(dtype=np.int16)
        #print(col_one_arr)
        col_one_arr_max = df['x'].max()
        row_one_arr_max = df['y'].max()
        col_one_arr_min = df['x'].min()
        row_one_arr_min = df['y'].min()
        z_array = np.zeros((col_one_arr_max-col_one_arr_min+1,row_one_arr_max-row_one_arr_min+1),dtype=np.uint8)
        z_array[col_one_arr-col_one_arr_min, row_one_arr-row_one_arr_min] = 1
        startTime2 = datetime.now()
        cached_array = maximalSquare(z_array)
        maxindex = cached_array.max()
        print(datetime.now() - startTime2)
        pozycja = unravel_index(cached_array.argmax(axis=None), (cached_array.shape))

        max_sqr_record = {
            'max_sqr_size':maxindex, 'max_sqr_x': col_one_arr_min+pozycja[0]-1, 'max_sqr_y': row_one_arr_min+pozycja[1]-1
        }
        Lista_max_square = Lista_max_square.append(max_sqr_record, ignore_index=True)
        print("Max_square size: "+str(maxindex) +" Corner data: " + str(col_one_arr_min+pozycja[0])+", "+str(row_one_arr_min+pozycja[1]))

        print("Ilość wczytanych kratek: " + str(len(polaczone_uzyt)))
        print("Ilość unikalnych kratek: " + str(len(unikalne_uzyt)))
        tuple_global += unikalne_uzyt

    print("Przetwarzanie unikalnych kratek od wszystkich:")
    print("Ilość wczytanych kratek: " + str(len(tuple_global)))
    #print("Ilość unikalnych kratek: " + str(len(unikalne_uzyt)))
        
    dane_global = pd.DataFrame(tuple_global, columns=['x', 'y','uid'])
    #print("Original Dataframe", dfObj, sep='\n')
    #print('*** Find Duplicate Rows based on all columns ***')

   

    
    dane_global = dane_global.groupby(["x","y"])['uid'].apply(list).reset_index()
    print(datetime.now() - startTime)
    dane_global.to_json('kwadraty_mapa.json', orient='records',indent=1)
    #print(uid_data)

    #przeniesc sortowanie linkow na poczatek bo trzeba 2 raazy skrypt odpalac
    uid_data = uid_data.drop(columns=['max_sqr_size','max_sqr_x','max_sqr_y'])
    #print(uid_data)
    result = pd.concat([uid_data, Lista_max_square.astype('Int64')], axis=1, sort=False)
    result = result.sort_values(by=['nick'], ascending=True).drop(columns=['uid'])
    #result.reset_index(drop=True,inplace=True)
    result.index = result.index + 1
    result['uid'] = result.index
    print(result)
    result.to_json('linki.json', orient='records',indent=1)
    #print(Lista_max_square)
    #print (dane_global)

except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')
