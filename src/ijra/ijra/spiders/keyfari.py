# -*- coding: utf-8 -*-
import scrapy

#https://ara.jri.ac.ir/Judge/Index?_IsOfficial=false&_IsCivil=false&_IsPenal=true&PageNumbers=1&PageNumber=1&PageSize=1000 کیفری

class KeyfariSpider(scrapy.Spider):
    name = 'kayfari'
    # allowed_domains = ['http://ara.jri.ac.ir/Judge/Index']
    start_urls = ['https://ara.jri.ac.ir/Judge/Index?_IsOfficial=false&_IsCivil=false&_IsPenal=true&PageNumbers=1&PageNumber=1&PageSize=1000']
    page = 2
    Ara=[]
    phase=0
    tak=0
    def parse(self, response):
        if KeyfariSpider.phase==0 :
            links = response.css("a::attr(href)").extract()
            sty = response.css("a").extract()
            for i in range(len(links)):
                if 'Judge/Text' in links[i] and 'Width:100' in sty[i]:
                    KeyfariSpider.Ara.append("https://ara.jri.ac.ir"+links[i])
            next_page = 'https://ara.jri.ac.ir/Judge/Index?_IsOfficial=false&_IsCivil=false&_IsPenal=true&PageNumbers=%s&PageNumber=1&PageSize=1000' % KeyfariSpider.page
            KeyfariSpider.page+=1
            if KeyfariSpider.page <5:
                yield scrapy.Request(next_page, callback=self.parse)
            else:
                KeyfariSpider.phase=1
                KeyfariSpider.page=0

                print(KeyfariSpider.Ara)
                print(len(KeyfariSpider.Ara))
                yield scrapy.Request(KeyfariSpider.Ara[0], callback=self.parse)
        else:
            try:
                dates = response.xpath("/html/body/section[3]/div[1]/div[2]/div/form/div/table//td[2]/text()[4]").extract()[0].strip()
                print(dates)
                try :
                    raay = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[1]/text()").extract()[0]
                    # print(raay)
                    try :
                        raay2 = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/div[1]/text()").extract()[0]
                        # print(raay)
                        if len(raay2)<50:
                            raay2 = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[3]/text()[1]").extract()[0]
                        if len(raay2) < 50:
                            raay2 = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/p[2]/text()[1]").extract()[0]
                        tak=0
                    except:
                        print("exception")
                        tak=1
                except:
                    tak=1
                    raay = response.xpath("/html/body/section[3]/div[2]/div/div/form/div/div[1]/text()").extract()[0]
                    print("except")
                    # input("comply?")
                if tak == 0:
                    yield{
                        "id": KeyfariSpider.page,
                        "date":dates,
                        "link":KeyfariSpider.Ara[KeyfariSpider.page],
                        "raay":raay,
                        "raay2":raay2
                    }
                else:
                    yield {
                        "id": KeyfariSpider.page,
                        "date": dates,
                        "link": KeyfariSpider.Ara[KeyfariSpider.page],
                        "raay": raay,
                        "raay2": "Not Available."
                    }
                KeyfariSpider.page += 1
                # input("comply?")
                if KeyfariSpider.page < len(KeyfariSpider.Ara):
                    yield scrapy.Request(KeyfariSpider.Ara[KeyfariSpider.page], callback=self.parse)
            except:
                print("Skipppppppppppppppppppppppppppinnnnnnnnnnnnngggggggggggggggggggggggg")
                KeyfariSpider.page+=1
                # input("comply?")
                if KeyfariSpider.page < len(KeyfariSpider.Ara) :
                    yield scrapy.Request(KeyfariSpider.Ara[KeyfariSpider.page], callback=self.parse)
