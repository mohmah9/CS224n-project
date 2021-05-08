# -*- coding: utf-8 -*-
import scrapy

#https://ara.jri.ac.ir/Judge/Index?_IsOfficial=false&_IsCivil=true&_IsPenal=false&PageNumbers=1&PageNumber=1&PageSize=1000 حقوقی
class HoghoghiSpider(scrapy.Spider):
    name = 'hoghoghi'
    # allowed_domains = ['http://ara.jri.ac.ir/Judge/Index']
    start_urls = ['https://ara.jri.ac.ir/Judge/Index?_IsOfficial=false&_IsCivil=true&_IsPenal=false&PageNumbers=1&PageNumber=1&PageSize=1000']
    page = 2
    Ara=[]
    phase=0
    def parse(self, response):
        if HoghoghiSpider.phase==0 :
            links = response.css("a::attr(href)").extract()
            sty = response.css("a").extract()
            for i in range(len(links)):
                if 'Judge/Text' in links[i] and 'Width:100' in sty[i]:
                    HoghoghiSpider.Ara.append("https://ara.jri.ac.ir"+links[i])
            next_page = 'https://ara.jri.ac.ir/Judge/Index?_IsOfficial=false&_IsCivil=true&_IsPenal=false&PageNumbers=%s&PageNumber=1&PageSize=1000' % HoghoghiSpider.page
            HoghoghiSpider.page+=1
            if HoghoghiSpider.page <7:
                yield scrapy.Request(next_page, callback=self.parse)
            else:
                HoghoghiSpider.phase=1
                HoghoghiSpider.page=0
                print(HoghoghiSpider.Ara)
                print(len(HoghoghiSpider.Ara))
                yield scrapy.Request(HoghoghiSpider.Ara[0], callback=self.parse)
        else:
            try:
                dates = response.xpath("/html/body/section[3]/div[1]/div[2]/div/form/div/table//td[2]/text()[4]").extract()[0].strip()
                print(dates)
                try:
                    raay = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[1]/text()").extract()[0]
                    if len(raay)<100:
                        print("fjkisdhfuhbfcobvdbcoubdhcubiujebve")
                        raay = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[3]/text()").extract()[0]
                        try:
                            raay2 = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[6]/text()").extract()[0]
                            tak=0
                        except:
                            tak=1
                    else:
                        try:
                            raay2 = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/div[1]/text()").extract()[0]
                            # print(raay)
                            if len(raay2) < 50:
                                raay2 = \
                                response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[3]/text()[1]").extract()[0]
                            if len(raay2) < 50:
                                raay2 = \
                                response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[2]/text()[1]").extract()[0]
                            tak = 0
                        except:
                            print("exception")
                            tak = 1
                except:
                    tak = 1
                    raay = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/div[1]/text()").extract()[0]
                    print("except")
                    # input("comply?")
                if tak == 0:
                    yield {
                        "id": HoghoghiSpider.page,
                        "date": dates,
                        "link": HoghoghiSpider.Ara[HoghoghiSpider.page],
                        "raay": raay,
                        "raay2": raay2
                    }
                else:
                    yield {
                        "id": HoghoghiSpider.page,
                        "date": dates,
                        "link": HoghoghiSpider.Ara[HoghoghiSpider.page],
                        "raay": raay,
                        "raay2": "Not Available."
                    }
                HoghoghiSpider.page += 1
                # input("comply?")
                if HoghoghiSpider.page < len(HoghoghiSpider.Ara):
                    yield scrapy.Request(HoghoghiSpider.Ara[HoghoghiSpider.page], callback=self.parse)
            except:
                print("Skipppppppppppppppppppppppppppinnnnnnnnnnnnngggggggggggggggggggggggg")
                HoghoghiSpider.page += 1
                # input("comply?")
                if HoghoghiSpider.page < len(HoghoghiSpider.Ara):
                    yield scrapy.Request(HoghoghiSpider.Ara[HoghoghiSpider.page], callback=self.parse)
