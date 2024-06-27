from selenium import webdriver
import datetime
import time


class TrajDownloader:
    def __init__(self, start, end, anums, n_days=180, save_path=r"assets"):
        self.start = start
        self.end = end
        self.anums = anums
        self.n_days = n_days
        option = webdriver.ChromeOptions()
        prefs = {
	        'download.default_directory': save_path,  # 设置默认下载路径
	        "profile.default_content_setting_values.automatic_downloads": 1  # 允许多文件下载
        }
        option.add_experimental_option("prefs", prefs)
        self.driver = webdriver.Edge(edge_options=option)
        self.today_time = int(time.mktime(time.strptime(str(datetime.date.today()), '%Y-%m-%d')))

    def download(self):
        for anum in self.anums:
            for i in range(self.n_days):
                t = self.today_time - 86400 * i
                url = "https://flightadsb.variflight.com/track-data/" + anum + "/" + t.__str__() + "000"
                self.driver.get(url)

                while True:
                    try:
                        self.driver.find_element("xpath", "//div[text()='No Data']")
                        break
                    except:
                        try:
                            if self.driver.find_element("xpath", "//div[@class='flight_status gray']").text == "到达":
                                if self.start in self.driver.find_element("xpath",
                                                       "(//span[@class='apt-name ellipsis'])[1]").text and self.end in self.driver.find_element(
                                                        "xpath","(//span[@class='apt-name ellipsis'])[2]").text:
                                    self.driver.find_element("xpath",
                                                        "//div[@id='app']/div[1]/div[1]/div[1]/div[3]/div[2]/div[2]/div[1]/div[1]/div[2]/div[1]/div[1]/div[" \
                                                        "1]/div[3]/table[1]/tbody[1]/tr[1]/td[11]/div[1]/div[1]/div[2]/a[1] ").click()
                                else:
                                    return
                            break
                        except:
                            pass
                    time.sleep(0.2)

if __name__ == '__main__':
    SHA_CAN = ['HO1855','MF7952','MU3852','ZH5238','GJ3888','HU7332','FM9303','MU8371','3U1318','3U1318',
               'CZ3596','CZ3596','HO7439','HO7439','MF1396','MF1396','MU5309','3U1280','3U1280','CZ3548',
               'CZ3548','HO7415','HO7415','MF1351','MF1351','MF3504','MU5301','3U1270','CZ3534','MF1337',
               'HO1857','MF7954','MU3854','ZH5240','3U5147','HO5422','MF3506','MU5303','HU7132','3U1264',]
    CAN_CTU = ['3U1163','CZ3401','HO7385','MF1237','3U1195','CZ3437','CZ3437','GJ5237','GJ5237','HO7389',
               'MF1261','3U8734','3U8734','CZ9558','HO5863','MF5558','MU3464','CZ7008','EU2236','TV5108',
               '3U1201','CZ3443','CZ3443','GJ5239','GJ5239','HO7391','MF1264','CA4306','TV6264','ZH4404',
               '3U1225','CZ3471','HO7393','MF1285','CA4320','3U8732','3U8732','CZ9556','G58726','HO5861',]
    GJS_CTU = ['3U8890','3U8890','CZ9616','GJ3172','MF5616','CA4182','TV6260','ZH4358','CA1405','TV6219',
               'ZH1405','CA4184','CA4184','CA1421','ZH1421','CA4186','ZH4362','TV6266','ZH4360','ZH4360',
               'CA1415','ZH1415','CA4120','ZH4328','3U8884','CZ9610','G56232','HO5889','MF5610','MU3508',
               'KN6192','MU6645','CA4114','ZH4322','3U1637','CA4114','CZ6183','MF1924','ZH4322','3U4384',]
    BJS_SIA = ['ZH1231','GS5137','HU7137','CA1231','SC5337','CA791','SC5353','TV6201','CA1231','SC5337',
               'TV6201','ZH1231','CA1289','SC5339','ZH1289','CA1289','SC5339','ZH1289','CA1209','SC5341',
               'TV6209','ZH1209','HU7237','CA1223','SC5343','ZH1223','CA1201','G58190','SC5345','ZH1201',
               'CA1201','G58190','SC5345','ZH1201','CA1225','G58141','SC5347','ZH1225','CA1229','SC5349',]
    downloader = TrajDownloader('上海','广州',SHA_CAN)
    downloader.download()