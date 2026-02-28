### သင်တန်းသားများလေ့ကျင့်ရန်

#### Backpropagation, NN Training, Improving, Torch, Tensor

- [[Fashion MNIST (Kaggle)]](https://www.kaggle.com/datasets/zalando-research/fashionmnist): အဝတ်အထည်ပုံရိပ်များကို Pixel value များဖြင့် ဖော်ပြထားသော CSV ဖြစ်သည်။ MNIST ကဲ့သို့ပင် ANN ဖြင့် ရှူးဖိနပ်၊ အင်္ကျီ စသည်တို့ကို ခွဲခြားရန် အလွန်သင့်တော်ပါသည်။ Manual testing အနေဖြင့် Pixel data အသစ်များကို model ထဲထည့်ပြီး အဖြေမှန်ကို စစ်ဆေးနိုင်သည်။

- [[Sign Language MNIST (Kaggle)]](https://www.kaggle.com/datasets/datamunge/sign-language-mnist): လက်သင်္ကေတပြ ဘာသာစကားပုံရိပ်များကို $28 \times 28$ pixel များအဖြစ် ပြောင်းလဲထားသော CSV ဖြစ်သည်။ ANN သုံးပြီး အက္ခရာများကို ခွဲခြားရန် အကောင်းဆုံးဖြစ်သည်။ ပုံရိပ်အသစ်များမှ pixel များကို manual test လုပ်ရန် ကောင်းမွန်သည်။

- [[A-Z Handwritten Alphabets (Kaggle)]](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format): အင်္ဂလိပ်အက္ခရာ A မှ Z အထိ လက်ရေးစာလုံးများကို $28 \times 28$ pixel value များဖြင့် စုစည်းထားသော CSV ဖြစ်သည်။ ANN ဖြင့် ပုံရိပ်များကို စာလုံးအဖြစ် ပြောင်းလဲပေးသည့် OCR စနစ်ငယ်တစ်ခုကို Manual testing လုပ်ကြည့်ရန် အကောင်းဆုံးဖြစ်သည်။

#### နမူနာ code & လေ့လာနည်းများ

- [[ MNIST Handwritten Digits ]](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv): လက်ရေးဂဏန်း (၀ မှ ၉) ပုံရိပ်များကို $28 \times 28$ (၇၈၄ ကော်လံ) pixel value များအဖြစ် ပြောင်းလဲထားသော CSV ဖြစ်သည်။ ANN (MLP/CNN) model များကို training ပေးရန်နှင့် validation ပြုလုပ်ရန် အခြေခံအကျဆုံး dataset ဖြစ်သည်။ CSV format ဖြစ်သဖြင့် Row တစ်ခုချင်းစီကို model ထဲထည့်သွင်းကာ Manual testing လုပ်ရန် အလွန်လွယ်ကူပါသည်။