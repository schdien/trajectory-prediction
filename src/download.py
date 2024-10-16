from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
import datetime
import time
import os

class AirlineInfo:
    def __init__(self,start,end,anums,label=None):
        self.start = start
        self.end = end
        self.anums = anums
        self.label = label


class TrajDownloader:
    def __init__(self, airlines:list[AirlineInfo], n_days=350):
        self.airlines = airlines
        self.n_days = n_days
        self.empty_counts = 0
        #option = webdriver.EdgeOptions()
        #option.add_experimental_option("prefs", {"download.default_directory": save_path})# 设置默认下载路径
        self.driver = webdriver.Edge()
        self.today_time = int(time.mktime(time.strptime(str(datetime.date.today()), '%Y-%m-%d')))
        self.xpath = {'no data':"//div[text()='No Data']", 'status':"//div[@class='flight_status gray']",
                      'start': "(//span[@class='apt-name ellipsis'])[1]", 'end':"(//span[@class='apt-name ellipsis'])[2]",
                      'download':"//div[@id='app']/div[1]/div[1]/div[1]/div[3]/div[2]/div[2]/div[1]/div[1]/div[2]/div[1]/div[1]/div[" \
                                                    "1]/div[3]/table[1]/tbody[1]/tr[1]/td[11]/div[1]/div[1]/div[2]/a[1] "}

    def download_airlines(self):
        for airline in self.airlines:
            self.download_airline(airline)

    def download_airline(self,airline:AirlineInfo):
        save_path = os.getcwd()+'\\assets' + '\\' + airline.label
        self.driver.execute_cdp_cmd('Browser.setDownloadBehavior',{'behavior':'allow','downloadPath':save_path})
        for anum in airline.anums:
            self.download_anum(airline.start,airline.end,anum)

    def download_anum(self,start,end,anum):
        for i in range(self.n_days):
            if self.empty_counts == 10:
                self.empty_counts = 0
                return
            t = self.today_time - 86400 * (i+1)
            url = "https://flightadsb.variflight.com/track-data/" + anum + "/" + t.__str__() + "000"
            self.driver.get(url)

            while True:
                try:
                    self.driver.find_element("xpath", self.xpath['no data'])
                    self.empty_counts += 1
                    return
                except:
                    try:
                        if self.driver.find_element("xpath", self.xpath['status']).text == "到达":
                            if start in self.driver.find_element("xpath", self.xpath['start']).text and end in self.driver.find_element("xpath",self.xpath['end']).text:
                                self.driver.find_element("xpath",self.xpath['download']).click()
                            else:
                                return
                        break
                    except:
                        pass
                time.sleep(0.1)

    def get_download_link(self,start,end,anum):
        for i in range(self.n_days):
            if self.empty_counts == 10:
                self.empty_counts = 0
                return
            t = self.today_time - 86400 * (i+1)
            url = "https://flightadsb.variflight.com/track-data/" + anum + "/" + t.__str__() + "000"
            self.driver.get(url)

            while True:
                try:
                    self.driver.find_element("xpath", self.xpath['no data'])
                    self.empty_counts += 1
                    return
                except:
                    try:
                        if self.driver.find_element("xpath", self.xpath['status']).text == "到达":
                            if start in self.driver.find_element("xpath", self.xpath['start']).text and end in self.driver.find_element("xpath",self.xpath['end']).text:
                                self.driver.find_element("xpath",self.xpath['download']).click()
                            else:
                                return
                        break
                    except:
                        pass
                time.sleep(0.01)
        

if __name__ == '__main__':
    #济南-武汉 done done
    #上海-广州 done done
    #上海-武汉 done
    #成都-广州 done
    #成都-上海 done
    #成都-武汉
    #北京-上海 done done
    #北京-西安 done
    #北京-武汉
    #北京-广州
    #广州-武汉 done



    #广州-成都
    anums = ['3U1163','CZ3401','HO7385','MF1237','3U1195','CZ3437','GJ5237','HO7389',
               'MF1261','3U8734','3U8734','CZ9558','HO5863','MF5558','MU3464','CZ7008','EU2236','TV5108',
               '3U1201','CZ3443','GJ5239','HO7391','MF1264','CA4306','TV6264','ZH4404',
               '3U1225','CZ3471','HO7393','MF1285','CA4320','3U8732','CZ9556','G58726','HO5861']
    can_ctu = AirlineInfo("广州","成都",anums,"CAN-CTU")

    #广州-武汉
    anums = ['MF3896','MU6733','CZ3705','CZ3705','CZ6223','CZ6608','CZ6608','MF1958','MF4190','CZ3348',
               'MF1202','CZ8672','MF4622','CZ5770','MF1783','MF3290','MU2541','CA8232','ZH4710','CZ3346',
               'CZ3346','CZ3346','MF1200','MF1200','CZ6589','CZ6589','CZ6589','CZ8730','MF4641','CZ8730',
               'MF4641']
    can_wuh = AirlineInfo("广州","武汉",anums,"CAN_WUH")

    #北京-成都
    anums = ['3U8890','3U8890','CZ9616','GJ3172','MF5616','CA4182','TV6260','ZH4358','CA1405','TV6219',
               'ZH1405','CA4184','CA4184','CA1421','ZH1421','CA4186','ZH4362','TV6266','ZH4360','ZH4360',
               'CA1415','ZH1415','CA4120','ZH4328','3U8884','CZ9610','G56232','HO5889','MF5610','MU3508',
               'KN6192','MU6645','CA4114','ZH4322','3U1637','CA4114','CZ6183','MF1924','ZH4322','3U4384',]
    bjs_ctu = AirlineInfo("北京","成都",anums,"BJS_CTU")

    #北京-西安
    anums = ['ZH1231','GS5137','HU7137','CA1231','SC5337','CA791','SC5353','TV6201','CA1231','SC5337',
               'TV6201','ZH1231','CA1289','SC5339','ZH1289','CA1289','SC5339','ZH1289','CA1209','SC5341',
               'TV6209','ZH1209','HU7237','CA1223','SC5343','ZH1223','CA1201','G58190','SC5345','ZH1201',
               'CA1201','G58190','SC5345','ZH1201','CA1225','G58141','SC5347','ZH1225','CA1229','SC5349',]
    bjs_sia = AirlineInfo("北京","西安",anums,"BJS_SIA")
    
    #上海-武汉
    anums = ['MF3261','MU2510','HO5362','MU2524','HO5358','MU2508','HO5356','MF3258','MU2506','CZ3824',
               'MF1569','HO5356','MF3258','MU2506','HO5354','MU2469','HO5364','MF3277','MU2534','HO5360',
               'MF3270','MU2522','CZ3580','HO7431','MF1382','CZ6172','MF1915','FM9363','MU8409','MF3262',
               'MU2512','CZ3544','MF1347','MF3264','MU2514','MF3256','MU2504']
    sha_wuh = AirlineInfo("上海","武汉",anums,"SHA_WUH")

    #上海-成都
    anums = ['CA9847','HO5478','MU5401','CA8547','ZH4831','HO5480','MU5405','MU5403','CZ9660','MF5660',
               'MU3556','CZ9660','MF5660','MU3556','CA4504','ZH4534','MU5411','CZ9654','HO5921','MF5654',
               'MU3550','MU3550','CA4516','ZH4546','CZ9719','EU3076','TV5366','HO5482','MU5409','CZ9891',
               'EU6672','TV5322','CA4240','ZH4388','CA4508','ZH4538','CA4508','ZH4538','CA3968','EU7844',
               'GJ5526','KY6882','SC6126','TV9882','ZH3928','CA8551','ZH4835','HO5484','MU5413','CA8549',
               'HO5755','ZH4833','CA3964']
    sha_ctu = AirlineInfo("上海","成都",anums,"SHA_CTU")


    airlines = [can_ctu,can_wuh,bjs_ctu,bjs_sia,sha_wuh,sha_ctu]
    downloader = TrajDownloader(airlines)
    downloader.download_airlines()