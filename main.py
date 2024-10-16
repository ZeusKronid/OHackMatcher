from datetime import date, datetime, timedelta
from typing import List
import holidays
import time

from papi_sdk import APIv3
from papi_sdk.models.search.base_request import GuestsGroup
from papi_sdk.models.search.region.b2b import B2BRegionRequest, B2BRegionResponse

import pandas as pd 
import numpy as np

import requests

import openmeteo_requests

import requests_cache
from retry_requests import retry

from bs4 import BeautifulSoup
from fake_useragent import UserAgent 

import re

import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#Download model from HuggingFaceAPI
#model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#model.save('MiniLM')

#Download ostrovok data from ostrovok API(data was transformed into DataFrames with another function which not implemented here)
#get  https://ostrovok.tech/hackathon/data/track_1/hotels_dumps.jsonl.

class YandexPriceParser:
    '''
        All interference with class is performed through self.parse method
        input: Ostrovok_matrix, checkin, checkout, number_of_adults
        output: Ostrovok matrix + yandex_price + yandex_name + ostrovok_price-yandex_price + ostrovok_price-yandex_price/ostrovok_price
        Access to output is performes with self.comparative_price_matrix
    '''
    
    def __init__(self, ostrovok_price_matrix, checkin, checkout, region_id, adults = 2, path_to_transformer = 'MiniLM'): 

        #Input
        self.ostrovok_price_matrix = ostrovok_price_matrix

        #Info for scraping
        self.checkin_date = checkin
        self.checkout_date = checkout
        self.adults=adults

        self.region_id = region_id
        self.region_name_dict = {
            1913 : 'krasnodar',
            1361 : 'nizhny-novgorod',
            2427 : 'minsk',
            481 : 'antalya-antalya-tr', 
            1178 : 'yerevan'
        }

        # Used to uniform meal names among hotels
        self.table_of_meal ={  
            
                'Без питания': 'nomeal',
                'Включён завтрак': 'breakfast', 
                'Включён завтрак и обед': 'half-board-lunch',
                'Включён завтрак и ужин' : 'half-board-dinner',
                'Полупансион' : 'half-board-dinner',
                'Включён завтрак, обед и ужин' : 'full-board',
                'Включён обед и ужин': 'full-board',
                'Включён обед' : 'lunch',
                'Включён ужин' : 'dinner',
        }
        
        #To avoid CAPTCHA
        self.user_agent = UserAgent()

        #NLP model to compare room_names
        self.nlp_model = SentenceTransformer(path_to_transformer)

        # Used to avoid starting from begining after crash 
        self.names_of_parsed_hotels = [] 
        
        #Output matrix
        self.comparative_price_matrix = pd.DataFrame()
        self.comparative_price_matrix['checkin'] = self.checkin_date
        self.comparative_price_matrix['checkout'] = self.checkout_date

    def parse(self, ): 
        '''
            Iterate through hotel_names
            Fetch url from self._get_hotel_url method
            Fetch page_data from self._get_page_data method
            If page is not valid retry 
            If page is valid connecting ostrovok data with yandex data through self._merge_ostrovok_with_yandex method
            outputs: self.comparative_price_matrix with all yandex prices for choosen ostrovok_matrix
        '''
        # Взять датасет прайсов
        # получить по названиям данные по ценам с сайта
        # сравнить данные для каждого сайта 
        # сопоставить внутри 
        
        
        hotel_names = self.ostrovok_price_matrix['hotel_name'].unique()

        for name in hotel_names:
            if name not in self.names_of_parsed_hotels:
                time.sleep(random.randint(1,2))
                url = self._get_hotel_url(name, 0)
                print(url)
                time.sleep(random.randint(1,5))
                page_data = self._get_page_data(url, 0)
                if page_data is False: 
                    continue
                elif page_data.empty: 
                    continue
                    
                #get info about rooms in one hotel
                restricted_ostrovok_matrix = self.ostrovok_price_matrix.loc[self.ostrovok_price_matrix['hotel_name'] == name]
                restricted_yandex_matrix = page_data
    
                self.comparative_price_matrix = pd.concat([self.comparative_price_matrix, self._merge_ostrovok_with_yandex(restricted_ostrovok_matrix, restricted_yandex_matrix)], axis=0, ignore_index=True)
                self.comparative_price_matrix['index'] = self.comparative_price_matrix['index'].apply(lambda x: str(x))
                self.names_of_parsed_hotels.append(name) # to avoid repetition
                
                time.sleep(random.randint(1,2))



    

    def _get_proxies(self, ):
        '''
            Method requests proxie from api.proxyscrape.com 
            output list of proxies http and socks4(socks4 unimplemented)
        '''
        proxies_http = requests.get('https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&protocol=http&proxy_format=protocolipport&format=text&anonymity=Anonymous,Elite&timeout=20000')
        proxies_https = requests.get('https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&protocol=socks5,socks4&proxy_format=protocolipport&format=text&anonymity=Anonymous,Elite&timeout=20000') 
        
        text_http = proxies_http.text
        proxie_list_http = text_http.split('\r\n')
        random.shuffle(proxie_list_http)

        text_https = proxies_https.text
        proxie_list_https = text_https.split('\r\n')
        random.shuffle(proxie_list_https)
        
        return proxie_list_http, proxie_list_https


    def _get_hotel_url(self, hotel_name, number_until_skip_url): 
        '''
            Receive yandex hotel url by parsing google page with hotel_name
            number_until_skip_url - to avoid infinite recursion
        '''
        
        dict_of_city_names =      {     1913 : 'Краснодар',
            1361 : 'Нижний+Новгород',
            2427 : 'Минск',
            481 : 'Анталья', 
            1178 : 'Ереван'}

        url = f'https://www.google.com/search?q={hotel_name.replace('&','').replace(' ', '+')}+{dict_of_city_names[self.region_id]}+Яндекс+Путешествия'
        
        headers = {
                  'User-Agent': self.user_agent.random,
              }
        res = requests.get(url, headers=headers)
        
        soup = BeautifulSoup(res.content, 'lxml')
        
        links = soup.find_all('a')
        
        for link in links:
            href = link.get('href')
            if href and 'travel.yandex.ru/hotels/' in href:
                url = href
                break
        print(url)
        match = re.search(fr'/hotels/{self.region_name_dict[self.region_id]}(?:-[^/]+)?/([^/]+)/?$', url)

        if match:
            hotel_id = match.group(1)  # Получаем текст между слэшами
        else:
            print("Не удалось найти нужную часть URL.")
            number_until_skip_url += 1 
            if number_until_skip_url >= 10:
                return False
            else:
                time.sleep(random.randint(5,10))
                return self._get_hotel_url(hotel_name, number_until_skip_url)

        final_url = f'https://travel.yandex.ru/hotels/{self.region_name_dict[self.region_id]}/{hotel_id}/?adults={self.adults}&checkinDate={self.checkin_date}&checkoutDate={self.checkout_date}&childrenAges=&searchPagePollingId=44ba2c70ebb1019ade557748e403a676-1-newsearch&seed=portal-hotels-search'
        return final_url 
        


    def _get_page_data(self, url, number_until_skip_page):

        '''
            Receive url, outputs dataframe with [room_name], ['meal'], ['price']
            number_until_skip_page - to avoid infinite recursion
        '''

        #Case when _get_hotel_url method was unable to find valid url
        if not url:
            return False
            
        #Init proxie and page-df to return
        proxie_list_http, proxie_list_https = self._get_proxies()
        page_df = pd.DataFrame()

        #Start rotate through proxies (https unimplemented)
        for i in range(0, len(proxie_list_http)):
            if i == len(proxie_list_http):
                return False
            print(proxie_list_http[i])
            headers = {
                'User-Agent': self.user_agent.random,
            }

            proxie_http = {'http': proxie_list_http[i]}
            
            res = requests.get(url, headers=headers, proxies = proxie_http)
            
            soup = BeautifulSoup(res.content, 'lxml')

            #No answer from peer
            if res.status_code != 200:
               # print('No answer from peer retrying with new proxie')
                time.sleep(random.randint(1,5))
                continue
            #CAPTCHA
            elif soup.title.string == 'Вы не робот?' or soup.title.string == 'Вы не робот? ':
               # print('CAPTCHA retrying with new proxie')
                time.sleep(random.randint(10,15)) # Best for avoiding
                number_until_skip_page +=1 
                if number_until_skip_page >15: 
                    time.sleep(random.randint(60,100))
                    return False
                
                continue
            else:
                break
        
        print(soup.title.string)

        #Return false if no rooms available
        rooms_available = soup.find_all('span', class_='a-NCA PwvPC Z7syr') 
        rooms_available = [i.text for i in rooms_available]
        
        if 'Невозможно забронировать' in rooms_available or f'На выбранные даты нет предложений для {self.adults} взрослых' in rooms_available: #
            #print('Нет номеров')
            return False

        rooms_available = soup.find_all('span', class_='GKiMq -uF7l Z7syr') 
        rooms_available = [i.text for i in rooms_available]
        if 'К сожалению, именно на эти даты и ваше число гостей свободных номеров нет' in rooms_available:
            #print('Нет номеров')
            return False

        #Gathering divs with info about rooms
        rooms_info = soup.find_all('div', class_='Bugf1 info_desktop') 
        
        list_of_dfs = []
        try:
            for room in rooms_info: 
                name = room.find('h3').text
                meals = room.find_all('span', class_='UzfXr TQ2-5 i9Gsh') 
                prices = room.find_all('span', class_='bQcBE G3B4k')
                
                meals = [i.text for i in meals]
    
                prices = [float(i.text.replace('\u2006', '').replace('₽', '')) for i in prices]
                helper_df = pd.DataFrame()
                
                helper_df['meal'] = meals
                helper_df['meal'] = helper_df['meal'].replace(self.table_of_meal)
                helper_df = helper_df[helper_df['meal'].isin(self.table_of_meal.values())]
                
                helper_df['yandex_price'] = prices
                
                helper_df['yandex_name'] = name                
                list_of_dfs.append(helper_df)
                
        except ValueError:
            time.sleep(random.randint(1,5))
            #print('Meal or price shape error recheck')
            return False

        #Sometimes page is empty just by chance in this case need to retry 
        if not list_of_dfs: 
            time.sleep(random.randint(1,5))
            #print('page is empty retry')
            number_until_skip_page +=1 
            if number_until_skip_page >= 4: 
                return False
            else:
                return self._get_page_data(url, number_until_skip_page)
            
        page_df = pd.concat(list_of_dfs, ignore_index=True)
        return page_df


    def _merge_ostrovok_with_yandex(self, restricted_ostrovok_matrix, restricted_yandex_matrix):

        '''
            input: ostrovok_price matrix for particular hotel, yandex_matrix parsed from parse method. 
            Find cosine similarity between ostrovok naming and yandex naming
            Alias most probable results
            Merge two Frames on the basis of rooms names and meal types
            output: merged matrix
             
        '''
        
        
        ostrovok_naming = list(restricted_ostrovok_matrix['room_name'].unique())
        yandex_naming= list(restricted_yandex_matrix['yandex_name'].unique())
        aliases = {}
        
        '''
        For each ostrovok names
        Find all cosine similarity for ALL names
        find the most probable result 
        Assign aliases
        Remove this pare 
        Repeat for next name
        
        '''
        #TODO: please test this behaviour twice
        iterate_len = len(ostrovok_naming)
        for i in range(0, iterate_len):
            list_of_sims = []
            for ostrovok_name in ostrovok_naming:
                if not ostrovok_naming: 
                    break
                sim = []
                ostrovok_vector = self.nlp_model.encode(ostrovok_name)
                for yandex_name in yandex_naming:
                    yandex_vector = self.nlp_model.encode(yandex_name)
                    sim.append(cosine_similarity([ostrovok_vector], [yandex_vector])[0][0])

                list_of_sims.append(sim)

            list_of_max_sim = [max(i) for i in list_of_sims]
            list_of_max_index = [i.index(max(i)) for i in list_of_sims]

            #Assign to ostrovok word to find alias index with maximum similarity
            ostrovok_index = list_of_max_sim.index(max(list_of_max_sim))
            #Assign to yandex name index with max similarity
            yandex_index = list_of_max_index[ostrovok_index]
            
            yandex_alias = yandex_naming[yandex_index]
            aliases[ostrovok_naming[ostrovok_index]] = yandex_alias
            yandex_naming.remove(yandex_alias)
            ostrovok_naming.remove(ostrovok_naming[ostrovok_index])
        # One of the way to improve quality is to place treshhold on cosine sim but this require alot of experementation

        #Redirect value:key to replace items to add room_name to yandex_matrix
        aliases = {value: key for key, value in aliases.items()}

        restricted_yandex_matrix['room_name'] = restricted_yandex_matrix['yandex_name'].replace(aliases)

        #Левый?
        #Merge by room_name and by type_of meal
        merged_matrix = pd.merge(restricted_ostrovok_matrix, restricted_yandex_matrix, on=['room_name', 'meal'], how='left', suffixes=('_left', '_right'))

        #Find price difference and percentage in price difference
        merged_matrix['price_diff'] = merged_matrix['price']-merged_matrix['yandex_price'] 
        merged_matrix['percentage_price_diff'] = merged_matrix['price_diff']/merged_matrix['price']
        #Put treshhold to avoid mismatch from data
        merged_matrix.loc[abs(merged_matrix['percentage_price_diff']) > 0.20, 'percentage_price_diff'] = None
        return merged_matrix



def split_year_intervals(start_date):
    intervals_7_days = []
    intervals_14_days = []
    
    # Определяем конец года
    end_date = start_date + timedelta(days=365)
    
    # Разбиваем на промежутки по 7 дней
    current_date = start_date
    while current_date < end_date:
        interval_start = current_date.date()
        interval_end = min(current_date + timedelta(days=6), end_date).date()
        intervals_7_days.append((interval_start, interval_end))
        current_date += timedelta(days=7)
    
    # Разбиваем на промежутки по 14 дней
    current_date = start_date
    while current_date < end_date:
        interval_start = current_date.date()
        interval_end = min(current_date + timedelta(days=13), end_date).date()
        intervals_14_days.append((interval_start, interval_end))
        current_date += timedelta(days=14)
    
    
    return intervals_7_days, intervals_14_days    




if __name__ == "__main__":
    
    with open('config.txt', 'r') as file:
        dict_api = eval(file.read())
    engine = create_engine(dict_api['password_bd'])
    
    while True: 
        week_intervals, two_week_intervals = split_year_intervals(datetime.today())
        region_ids = [1913, 1361,2427, 1178,] #1639 481-- turkey out1913
        
        
        for region_id in region_ids:
            for interval in week_intervals: 
                checkin, checkout = interval
                try:
                    fetch = PricesFetcher(region_id, checkin, checkout)
                except Exception:
                    continue
                parser = YandexPriceParser(fetch.prices_ostrovok, str(checkin), str(checkout), region_id)
                try:
                    parser.parse()
                except Exception:
                    parser.comparative_price_matrix.to_sql('Hotel', engine, if_exists='replace', index=False) #dropna()
                    continue
                parser.comparative_price_matrix.to_sql('Hotel', engine, if_exists='replace', index=False) #dropna()
       
               # parser.comparative_price_matrix[].to_sql('Hotel', engine, if_exists='replace', index=False)
    
    
        time.sleep(28800) # 8 hours
