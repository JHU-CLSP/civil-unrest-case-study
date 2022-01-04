# Civil Unrest Case Study
Code from "Study of Manifestation of Civil Unrest on Twitter" [W-NUT @ EMNLP 2021](http://noisy-text.github.io/2021/)

---
## Data

### Global Civil Unrest on Twitter
The case studies and civil unrest models were trained on a new dataset introduced in the paper, **Global Civil Unrest on Twitter (G-CUT)**. The dataset is available for download [here](https://zenodo.org/record/5816218).

As per Twitter TOS, we do not supply the full tweet. We include:
* Tweet ID
* Creation date
* Country of origin (code in ISO Alpha 2, as specified in the tweet geotag)
* Country of origin (code in ISO Alpha 3 format)
* Civil unrest filtration model score as provided by the BERTweet model from [Sech et al 2020](https://github.com/AADeLucia/JHU-CUT)
* Civil unrest related: a convenience boolean whether the civil unrest filtration model score is >= 0.5

We suggest using a tool like [Hydrator](https://github.com/DocNow/hydrator) to re-collect the tweets.

Countries included in the dataset: 

Algeria, Angola, Bangladesh, Benin, Burkina Faso, Burundi, Cambodia, Cameroon, Central African Republic, Chad, Democratic Republic of Congo, Egypt, Ethiopia, Ghana, Guinea, Ivory Coast, Kenya, Liberia, Libya, Madagascar, Malawi, Mali, Morocco, Mozambique, Myanmar, Namibia, Nepal, Niger, Nigeria, Pakistan, Senegal, Sierra Leone, Somalia, South Africa, Sri Lanka, Sudan, Tanzania, Thailand, Togo, Tunisia, Uganda, Zambia


| Country | All / Filtered       | Country | All / Filtered         | Country | All / Filtered       |
|-----|------------------------|-----|------------------------|-----|------------------------|
| AGO | 289,373 / 16,639       | BDI | 25,041 / 6,097         | BEN | 150,348 / 16,489       |
| BFA | 50,197 / 4,327         | BGD | 2,318,969 / 330,947    | CAF | 28,723 / 2,860         |
| CIV | 564,197 / 34,286       | CMR | 833,795 / 71,420       | COD | 251,430 / 27,920       |
| DZA | 1,561,362 / 150,084    | EGY | 8,845,187 / 608,977    | ETH | 261,202 / 59,111       |
| GHA | 11,772,277 / 1,711,557 | GIN | 68,168 / 4,652         | KEN | 11,837,021 / 2,451,866 |
| KHM | 658,744 / 86,021       | LBR | 114,329 / 20,267       | LBY | 674,195 / 55,177       |
| LKA | 2,312,676 / 320,593    | MAR | 2,155,938 / 243,772    | MDG | 118,673 / 10,622       |
| MLI | 77,523 / 6,782         | MMR | 552,406 / 81,130       | MOZ | 349,321 / 31,634       |
| MWI | 780,767 / 101,891      | NAM | 1,881,238 / 238,720    | NER | 58,960 / 5,220         |
| NGA | 48,954,857 / 9,660,532 | NPL | 1,789,592 / 239,200    | PAK | 15,927,538 / 2,966,772 |
| SDN | 743,925 / 55,151       | SEN | 684,381 / 40,840       | SLE | 131,235 / 19,460       |
| SOM | 215,610 / 60,061       | TCD | 26,951 / 2,577         | TGO | 58,253 / 8,254         |
| THA | 14,661,060 / 980,846   | TUN | 944,833 / 87,903       | TZA | 2,216,871 / 248,856    |
| UGA | 3,274,687 / 638,372    | ZAF | 72,155,722 / 9,323,649 | ZMB | 1,706,438 / 246,815    |


### ACLED
The ground-truth ACLED dataset can be gathered from [https://acleddata.com/data-export-tool/](https://acleddata.com/data-export-tool/) with the following parameters:

* Event type: Riots and Protests
* Years: 2014 - 2019
* Countries: codes listed in the table above

We also include the data `/data/2014-01-01-2020-01-01_acled_reduced_all.csv`.

---
## Code
We have included the code and notebooks to reproduce our analyses and models in `/code` and `/notebooks`. The location debiasing features are in `/data/location_stopwords.txt` (combined with standard English stopwords).

---
If you have any questions, please contact Alexandra DeLucia via email [aadelucia@jhu.edu](mailto:aadelucia@jhu.edu)

