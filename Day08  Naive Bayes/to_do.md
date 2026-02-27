### သင်တန်းသားများလေ့ကျင့်ရန်


#### Naive Bayes (Advanced / real-world datasets)

- [[Language identification datasets]](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst): ဘာသာစကားတစ်ခုစီတွင် ပါဝင်သော စကားလုံးများ၏ Frequency ပေါ်မူတည်၍ Multinomial NB ဖြင့် Language Classification ပြုလုပ်နိုင်ပါသည်။ အကယ်၍ စာသားသည် အလွန်တိုတောင်းပြီး စကားလုံးပါဝင်မှု ရှိ/မရှိ (0/1) ကိုသာ ကြည့်လိုလျှင် Bernoulli NB ကို ပြောင်းလဲအသုံးပြုနိုင်ပါသည်။

- [[Bank Customer Churn Prediction]](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction): ဘဏ်ဖောက်သည်များ ဆက်လက်အသုံးပြုခြင်း ရှိ၊ မရှိကို ခန့်မှန်းရသည့် dataset ဖြစ်သည်။ Geography နှင့် Gender ကဲ့သို့သော Category features များအတွက် Categorical NB ကို လည်းကောင်း၊ အသက် (Age) နှင့် လက်ကျန်ငွေ (Balance) ကဲ့သို့သော Continuous features များအတွက် Gaussian NB ကို လည်းကောင်း ပေါင်းစပ်အသုံးပြုနိုင်ပါသည်။

- [[Spam Email Dataset]](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection): အီးမေးလ်များအတွင်းရှိ စကားလုံးများ၏ အကြိမ်ရေ (Word frequencies) ကို အခြေခံထားသည့် CSV dataset ဖြစ်သည်။ Multinomial Naive Bayes ကို အသုံးပြု၍ Spam နှင့် Ham (သာမန်အီးမေးလ်) ကို အမြန်ဆုံးနှင့် အတိကျဆုံး ခွဲခြားနိုင်ပါသည်။

- [[COVID-19 Patient Health Dataset]](https://www.kaggle.com/datasets/meirnizri/covid19-dataset): ဤ dataset တွင် လူနာများ၏ နောက်ခံရောဂါအခြေအနေများနှင့် ဆေးရုံတက်ရမှု အချက်အလက်များ ပါဝင်သည်။ ရောဂါလက္ခဏာ ရှိ/မရှိ (Binary features) များအတွက် Bernoulli NB ကို လည်းကောင်း၊ အသက် (Age) ကဲ့သို့သော Continuous data များအတွက် Gaussian NB ကို လည်းကောင်း အသုံးပြု၍ လူနာ၏ အခြေအနေ စိုးရိမ်ရမှု ရှိ/မရှိကို ခန့်မှန်းနိုင်ပါသည်

- [[ Mobile Phone Price Prediction ]](https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price): ဖုန်းများ၏ RAM၊ Storage၊ Screen Size နှင့် Camera specs များကဲ့သို့သော Continuous features များပါဝင်သည့် CSV dataset ဖြစ်သည်။ ကိန်းဂဏန်း တိုင်းတာချက်များ ဖြစ်သောကြောင့် Gaussian Naive Bayes ကို အသုံးပြု၍ ဈေးနှုန်းအဆင့်အတန်းကို ခန့်မှန်းရန် အထူးသင့်တော်ပါသည်။

- [[Car Features and MSRP]]():ဤ CSV dataset တွင် ကားအမှတ်တံဆိပ်၊ အင်ဂျင်အမျိုးအစားနှင့် ဈေးနှုန်းများစွာ ပါဝင်သည်။ ကားအမှတ်တံဆိပ် အရေအတွက် မမျှတမှု (Imbalance) ရှိသောကြောင့် Complement NB ဖြင့် ကားအမျိုးအစား ခွဲခြားရန် အကောင်းဆုံးဖြစ်သည်။(Horsepower,MPG,MSRP) ကားတစ်စီး၏ အချက်အလက်များကို ရိုက်ထည့်လိုက်ရုံဖြင့် ၎င်းသည် မည်သည့်ကားအမျိုးအစား (ဥပမာ - Luxury, Performance) ဖြစ်သည်ကို ခန့်မှန်းခိုင်းခြင်းဖြင့် Manual Testing လုပ်ရန် အလွန်စိတ်ဝင်စားဖို့ကောင်းပါသည်။

#### နမူနာ code

- [[Airline-sentiment-tweets]](https://www.kaggle.com/datasets/tango911/airline-sentiment-tweets): ဤ dataset သည် Twitter ပေါ်မှ US လေကြောင်းလိုင်းများနှင့် ပတ်သက်သည့် အသုံးပြုသူများ၏ လေကြောင်းလိုင်းအလိုက် ဝန်ဆောင်မှုများအပေါ် သဘောထားကို (positive)၊ (negative) သို့မဟုတ် (neutral) စသည်ဖြင့် Sentiment Analysis ပြုလုပ်ပြီး ခွဲခြားခန့်မှန်းရသည့် dataset တစ်ခု ဖြစ်ပါသည်။


#### အသုံးပြုရန်
- Model choices: `GaussianNB`, `MultinomialNB`, `BernoulliNB`,`ComplementNB`, `CategoricalNB` (scikit-learn)
