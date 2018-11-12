Titanic
================
Abby Bergman
11/6/2018

``` r
#Estimate three different logistic regression models with Survived as the response variable. You may use any combination of the predictors to estimate these models. Don’t just reuse the models from the notes.

#survival by fare
survive_fare <- glm(Survived ~ Fare, data = titanic_train, family = binomial)
summary(survive_fare)
```

    ## 
    ## Call:
    ## glm(formula = Survived ~ Fare, family = binomial, data = titanic_train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.4906  -0.8878  -0.8531   1.3429   1.5942  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -0.941330   0.095129  -9.895  < 2e-16 ***
    ## Fare         0.015197   0.002232   6.810 9.79e-12 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1186.7  on 890  degrees of freedom
    ## Residual deviance: 1117.6  on 889  degrees of freedom
    ## AIC: 1121.6
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
#survival by class
survive_class <- glm(Survived ~ Pclass, data = titanic_train, family = binomial)
summary(survive_class)
```

    ## 
    ## Call:
    ## glm(formula = Survived ~ Pclass, family = binomial, data = titanic_train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.4390  -0.7569  -0.7569   0.9367   1.6673  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  1.44679    0.20743   6.975 3.06e-12 ***
    ## Pclass      -0.85011    0.08715  -9.755  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1186.7  on 890  degrees of freedom
    ## Residual deviance: 1084.4  on 889  degrees of freedom
    ## AIC: 1088.4
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
#survival by sex
survive_sex <- glm(Survived ~ Sex, data = titanic_train, family = binomial)
summary(survive_sex)
```

    ## 
    ## Call:
    ## glm(formula = Survived ~ Sex, family = binomial, data = titanic_train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.6462  -0.6471  -0.6471   0.7725   1.8256  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)   1.0566     0.1290   8.191 2.58e-16 ***
    ## Sexmale      -2.5137     0.1672 -15.036  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1186.7  on 890  degrees of freedom
    ## Residual deviance:  917.8  on 889  degrees of freedom
    ## AIC: 921.8
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
#Calculate the leave-one-out-cross-validation error rate for each model. Which model performs the best?
#survive_fare===================================================#begin cross validation
loocv_data <- loo_cv(titanic_train)
loocv_data
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 891 x 2
    ##    splits       id        
    ##    <list>       <chr>     
    ##  1 <S3: rsplit> Resample1 
    ##  2 <S3: rsplit> Resample2 
    ##  3 <S3: rsplit> Resample3 
    ##  4 <S3: rsplit> Resample4 
    ##  5 <S3: rsplit> Resample5 
    ##  6 <S3: rsplit> Resample6 
    ##  7 <S3: rsplit> Resample7 
    ##  8 <S3: rsplit> Resample8 
    ##  9 <S3: rsplit> Resample9 
    ## 10 <S3: rsplit> Resample10
    ## # ... with 881 more rows

``` r
#

holdout_results <- function(splits){

  mod <- glm(Survived ~ Fare, data = analysis(splits)
             )  
  holdout <- assessment(splits)
  
  res <- augment(mod, newdata = holdout) %>%
    mutate(.resid = Survived -.fitted)
  res
}

#use map
loocv_data$results <- map(loocv_data$splits, holdout_results)
loocv_data$mse <- map_dbl(loocv_data$results, ~mean(.$.resid^2))
loocv_data
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 891 x 4
    ##    splits       id         results             mse
    ##    <list>       <chr>      <list>            <dbl>
    ##  1 <S3: rsplit> Resample1  <tibble [1 × 16]> 0.106
    ##  2 <S3: rsplit> Resample2  <tibble [1 × 16]> 0.438
    ##  3 <S3: rsplit> Resample3  <tibble [1 × 16]> 0.131
    ##  4 <S3: rsplit> Resample4  <tibble [1 × 16]> 0.286
    ##  5 <S3: rsplit> Resample5  <tibble [1 × 16]> 0.105
    ##  6 <S3: rsplit> Resample6  <tibble [1 × 16]> 0.139
    ##  7 <S3: rsplit> Resample7  <tibble [1 × 16]> 0.398
    ##  8 <S3: rsplit> Resample8  <tibble [1 × 16]> 0.460
    ##  9 <S3: rsplit> Resample9  <tibble [1 × 16]> 0.137
    ## 10 <S3: rsplit> Resample10 <tibble [1 × 16]> 0.144
    ## # ... with 881 more rows

``` r
#overall MSE
loocv_data %>%
  summarize(mse_fare = mean(mse))
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 1 x 1
    ##   mse_fare
    ##      <dbl>
    ## 1    0.222

The mean squared error for survival by fare is 0.222

``` r
#survive_class=====================================================
#begin cross validation
loocv_data <- loo_cv(titanic_train)
loocv_data
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 891 x 2
    ##    splits       id        
    ##    <list>       <chr>     
    ##  1 <S3: rsplit> Resample1 
    ##  2 <S3: rsplit> Resample2 
    ##  3 <S3: rsplit> Resample3 
    ##  4 <S3: rsplit> Resample4 
    ##  5 <S3: rsplit> Resample5 
    ##  6 <S3: rsplit> Resample6 
    ##  7 <S3: rsplit> Resample7 
    ##  8 <S3: rsplit> Resample8 
    ##  9 <S3: rsplit> Resample9 
    ## 10 <S3: rsplit> Resample10
    ## # ... with 881 more rows

``` r
#

holdout_results <- function(splits){

  mod <- glm(Survived ~ Pclass, data = analysis(splits)
             )  
  holdout <- assessment(splits)
  
  res <- augment(mod, newdata = holdout) %>%
    mutate(.resid = Survived -.fitted)
  res
}

#use map
loocv_data$results <- map(loocv_data$splits, holdout_results)
loocv_data$mse <- map_dbl(loocv_data$results, ~mean(.$.resid^2))
loocv_data
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 891 x 4
    ##    splits       id         results              mse
    ##    <list>       <chr>      <list>             <dbl>
    ##  1 <S3: rsplit> Resample1  <tibble [1 × 16]> 0.198 
    ##  2 <S3: rsplit> Resample2  <tibble [1 × 16]> 0.0616
    ##  3 <S3: rsplit> Resample3  <tibble [1 × 16]> 0.568 
    ##  4 <S3: rsplit> Resample4  <tibble [1 × 16]> 0.0616
    ##  5 <S3: rsplit> Resample5  <tibble [1 × 16]> 0.129 
    ##  6 <S3: rsplit> Resample6  <tibble [1 × 16]> 0.0616
    ##  7 <S3: rsplit> Resample7  <tibble [1 × 16]> 0.0616
    ##  8 <S3: rsplit> Resample8  <tibble [1 × 16]> 0.0616
    ##  9 <S3: rsplit> Resample9  <tibble [1 × 16]> 0.198 
    ## 10 <S3: rsplit> Resample10 <tibble [1 × 16]> 0.129 
    ## # ... with 881 more rows

``` r
#overall MSE
loocv_data %>%
  summarize(mse_class = mean(mse))
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 1 x 1
    ##   mse_class
    ##       <dbl>
    ## 1     0.210

The mean squared error rate for survival by class is 0.210.

``` r
#survive_sex======================================================
#begin cross validation
loocv_data <- loo_cv(titanic_train)

#

holdout_results <- function(splits){

  mod <- glm(Survived ~ Sex, data = analysis(splits)
             )  
  holdout <- assessment(splits)
  
  res <- augment(mod, newdata = holdout) %>%
    mutate(.resid = Survived -.fitted)
  res
}

#use map
loocv_data$results <- map(loocv_data$splits, holdout_results)
loocv_data$mse <- map_dbl(loocv_data$results, ~mean(.$.resid^2))
loocv_data
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 891 x 4
    ##    splits       id         results              mse
    ##    <list>       <chr>      <list>             <dbl>
    ##  1 <S3: rsplit> Resample1  <tibble [1 × 16]> 0.554 
    ##  2 <S3: rsplit> Resample2  <tibble [1 × 16]> 0.0670
    ##  3 <S3: rsplit> Resample3  <tibble [1 × 16]> 0.0670
    ##  4 <S3: rsplit> Resample4  <tibble [1 × 16]> 0.660 
    ##  5 <S3: rsplit> Resample5  <tibble [1 × 16]> 0.0358
    ##  6 <S3: rsplit> Resample6  <tibble [1 × 16]> 0.0358
    ##  7 <S3: rsplit> Resample7  <tibble [1 × 16]> 0.0358
    ##  8 <S3: rsplit> Resample8  <tibble [1 × 16]> 0.0670
    ##  9 <S3: rsplit> Resample9  <tibble [1 × 16]> 0.0358
    ## 10 <S3: rsplit> Resample10 <tibble [1 × 16]> 0.660 
    ## # ... with 881 more rows

``` r
#overall MSE
loocv_data %>%
  summarize(mse_sex = mean(mse))
```

    ## # Leave-one-out cross-validation 
    ## # A tibble: 1 x 1
    ##   mse_sex
    ##     <dbl>
    ## 1   0.167

The mean squared error rate for survival by sex is 0.167

``` r
#Now estimate three random forest models. Generate random forests with 500 trees apiece.
#convert qualitative to factors
titanic_tree_data <- titanic_train %>%
  mutate(Survived = if_else(Survived == 1, "Survived", "Died"),
         Survived = as.factor(Survived),
         Sex = as.factor(Sex))
titanic_tree_data
```

    ##     PassengerId Survived Pclass
    ## 1             1     Died      3
    ## 2             2 Survived      1
    ## 3             3 Survived      3
    ## 4             4 Survived      1
    ## 5             5     Died      3
    ## 6             6     Died      3
    ## 7             7     Died      1
    ## 8             8     Died      3
    ## 9             9 Survived      3
    ## 10           10 Survived      2
    ## 11           11 Survived      3
    ## 12           12 Survived      1
    ## 13           13     Died      3
    ## 14           14     Died      3
    ## 15           15     Died      3
    ## 16           16 Survived      2
    ## 17           17     Died      3
    ## 18           18 Survived      2
    ## 19           19     Died      3
    ## 20           20 Survived      3
    ## 21           21     Died      2
    ## 22           22 Survived      2
    ## 23           23 Survived      3
    ## 24           24 Survived      1
    ## 25           25     Died      3
    ## 26           26 Survived      3
    ## 27           27     Died      3
    ## 28           28     Died      1
    ## 29           29 Survived      3
    ## 30           30     Died      3
    ## 31           31     Died      1
    ## 32           32 Survived      1
    ## 33           33 Survived      3
    ## 34           34     Died      2
    ## 35           35     Died      1
    ## 36           36     Died      1
    ## 37           37 Survived      3
    ## 38           38     Died      3
    ## 39           39     Died      3
    ## 40           40 Survived      3
    ## 41           41     Died      3
    ## 42           42     Died      2
    ## 43           43     Died      3
    ## 44           44 Survived      2
    ## 45           45 Survived      3
    ## 46           46     Died      3
    ## 47           47     Died      3
    ## 48           48 Survived      3
    ## 49           49     Died      3
    ## 50           50     Died      3
    ## 51           51     Died      3
    ## 52           52     Died      3
    ## 53           53 Survived      1
    ## 54           54 Survived      2
    ## 55           55     Died      1
    ## 56           56 Survived      1
    ## 57           57 Survived      2
    ## 58           58     Died      3
    ## 59           59 Survived      2
    ## 60           60     Died      3
    ## 61           61     Died      3
    ## 62           62 Survived      1
    ## 63           63     Died      1
    ## 64           64     Died      3
    ## 65           65     Died      1
    ## 66           66 Survived      3
    ## 67           67 Survived      2
    ## 68           68     Died      3
    ## 69           69 Survived      3
    ## 70           70     Died      3
    ## 71           71     Died      2
    ## 72           72     Died      3
    ## 73           73     Died      2
    ## 74           74     Died      3
    ## 75           75 Survived      3
    ## 76           76     Died      3
    ## 77           77     Died      3
    ## 78           78     Died      3
    ## 79           79 Survived      2
    ## 80           80 Survived      3
    ## 81           81     Died      3
    ## 82           82 Survived      3
    ## 83           83 Survived      3
    ## 84           84     Died      1
    ## 85           85 Survived      2
    ## 86           86 Survived      3
    ## 87           87     Died      3
    ## 88           88     Died      3
    ## 89           89 Survived      1
    ## 90           90     Died      3
    ## 91           91     Died      3
    ## 92           92     Died      3
    ## 93           93     Died      1
    ## 94           94     Died      3
    ## 95           95     Died      3
    ## 96           96     Died      3
    ## 97           97     Died      1
    ## 98           98 Survived      1
    ## 99           99 Survived      2
    ## 100         100     Died      2
    ## 101         101     Died      3
    ## 102         102     Died      3
    ## 103         103     Died      1
    ## 104         104     Died      3
    ## 105         105     Died      3
    ## 106         106     Died      3
    ## 107         107 Survived      3
    ## 108         108 Survived      3
    ## 109         109     Died      3
    ## 110         110 Survived      3
    ## 111         111     Died      1
    ## 112         112     Died      3
    ## 113         113     Died      3
    ## 114         114     Died      3
    ## 115         115     Died      3
    ## 116         116     Died      3
    ## 117         117     Died      3
    ## 118         118     Died      2
    ## 119         119     Died      1
    ## 120         120     Died      3
    ## 121         121     Died      2
    ## 122         122     Died      3
    ## 123         123     Died      2
    ## 124         124 Survived      2
    ## 125         125     Died      1
    ## 126         126 Survived      3
    ## 127         127     Died      3
    ## 128         128 Survived      3
    ## 129         129 Survived      3
    ## 130         130     Died      3
    ## 131         131     Died      3
    ## 132         132     Died      3
    ## 133         133     Died      3
    ## 134         134 Survived      2
    ## 135         135     Died      2
    ## 136         136     Died      2
    ## 137         137 Survived      1
    ## 138         138     Died      1
    ## 139         139     Died      3
    ## 140         140     Died      1
    ## 141         141     Died      3
    ## 142         142 Survived      3
    ## 143         143 Survived      3
    ## 144         144     Died      3
    ## 145         145     Died      2
    ## 146         146     Died      2
    ## 147         147 Survived      3
    ## 148         148     Died      3
    ## 149         149     Died      2
    ## 150         150     Died      2
    ## 151         151     Died      2
    ## 152         152 Survived      1
    ## 153         153     Died      3
    ## 154         154     Died      3
    ## 155         155     Died      3
    ## 156         156     Died      1
    ## 157         157 Survived      3
    ## 158         158     Died      3
    ## 159         159     Died      3
    ## 160         160     Died      3
    ## 161         161     Died      3
    ## 162         162 Survived      2
    ## 163         163     Died      3
    ## 164         164     Died      3
    ## 165         165     Died      3
    ## 166         166 Survived      3
    ## 167         167 Survived      1
    ## 168         168     Died      3
    ## 169         169     Died      1
    ## 170         170     Died      3
    ## 171         171     Died      1
    ## 172         172     Died      3
    ## 173         173 Survived      3
    ## 174         174     Died      3
    ## 175         175     Died      1
    ## 176         176     Died      3
    ## 177         177     Died      3
    ## 178         178     Died      1
    ## 179         179     Died      2
    ## 180         180     Died      3
    ## 181         181     Died      3
    ## 182         182     Died      2
    ## 183         183     Died      3
    ## 184         184 Survived      2
    ## 185         185 Survived      3
    ## 186         186     Died      1
    ## 187         187 Survived      3
    ## 188         188 Survived      1
    ## 189         189     Died      3
    ## 190         190     Died      3
    ## 191         191 Survived      2
    ## 192         192     Died      2
    ## 193         193 Survived      3
    ## 194         194 Survived      2
    ## 195         195 Survived      1
    ## 196         196 Survived      1
    ## 197         197     Died      3
    ## 198         198     Died      3
    ## 199         199 Survived      3
    ## 200         200     Died      2
    ## 201         201     Died      3
    ## 202         202     Died      3
    ## 203         203     Died      3
    ## 204         204     Died      3
    ## 205         205 Survived      3
    ## 206         206     Died      3
    ## 207         207     Died      3
    ## 208         208 Survived      3
    ## 209         209 Survived      3
    ## 210         210 Survived      1
    ## 211         211     Died      3
    ## 212         212 Survived      2
    ## 213         213     Died      3
    ## 214         214     Died      2
    ## 215         215     Died      3
    ## 216         216 Survived      1
    ## 217         217 Survived      3
    ## 218         218     Died      2
    ## 219         219 Survived      1
    ## 220         220     Died      2
    ## 221         221 Survived      3
    ## 222         222     Died      2
    ## 223         223     Died      3
    ## 224         224     Died      3
    ## 225         225 Survived      1
    ## 226         226     Died      3
    ## 227         227 Survived      2
    ## 228         228     Died      3
    ## 229         229     Died      2
    ## 230         230     Died      3
    ## 231         231 Survived      1
    ## 232         232     Died      3
    ## 233         233     Died      2
    ## 234         234 Survived      3
    ## 235         235     Died      2
    ## 236         236     Died      3
    ## 237         237     Died      2
    ## 238         238 Survived      2
    ## 239         239     Died      2
    ## 240         240     Died      2
    ## 241         241     Died      3
    ## 242         242 Survived      3
    ## 243         243     Died      2
    ## 244         244     Died      3
    ## 245         245     Died      3
    ## 246         246     Died      1
    ## 247         247     Died      3
    ## 248         248 Survived      2
    ## 249         249 Survived      1
    ## 250         250     Died      2
    ## 251         251     Died      3
    ## 252         252     Died      3
    ## 253         253     Died      1
    ## 254         254     Died      3
    ## 255         255     Died      3
    ## 256         256 Survived      3
    ## 257         257 Survived      1
    ## 258         258 Survived      1
    ## 259         259 Survived      1
    ## 260         260 Survived      2
    ## 261         261     Died      3
    ## 262         262 Survived      3
    ## 263         263     Died      1
    ## 264         264     Died      1
    ## 265         265     Died      3
    ## 266         266     Died      2
    ## 267         267     Died      3
    ## 268         268 Survived      3
    ## 269         269 Survived      1
    ## 270         270 Survived      1
    ## 271         271     Died      1
    ## 272         272 Survived      3
    ## 273         273 Survived      2
    ## 274         274     Died      1
    ## 275         275 Survived      3
    ## 276         276 Survived      1
    ## 277         277     Died      3
    ## 278         278     Died      2
    ## 279         279     Died      3
    ## 280         280 Survived      3
    ## 281         281     Died      3
    ## 282         282     Died      3
    ## 283         283     Died      3
    ## 284         284 Survived      3
    ## 285         285     Died      1
    ## 286         286     Died      3
    ## 287         287 Survived      3
    ## 288         288     Died      3
    ## 289         289 Survived      2
    ## 290         290 Survived      3
    ## 291         291 Survived      1
    ## 292         292 Survived      1
    ## 293         293     Died      2
    ## 294         294     Died      3
    ## 295         295     Died      3
    ## 296         296     Died      1
    ## 297         297     Died      3
    ## 298         298     Died      1
    ## 299         299 Survived      1
    ## 300         300 Survived      1
    ## 301         301 Survived      3
    ## 302         302 Survived      3
    ## 303         303     Died      3
    ## 304         304 Survived      2
    ## 305         305     Died      3
    ## 306         306 Survived      1
    ## 307         307 Survived      1
    ## 308         308 Survived      1
    ## 309         309     Died      2
    ## 310         310 Survived      1
    ## 311         311 Survived      1
    ## 312         312 Survived      1
    ## 313         313     Died      2
    ## 314         314     Died      3
    ## 315         315     Died      2
    ## 316         316 Survived      3
    ## 317         317 Survived      2
    ## 318         318     Died      2
    ## 319         319 Survived      1
    ## 320         320 Survived      1
    ## 321         321     Died      3
    ## 322         322     Died      3
    ## 323         323 Survived      2
    ## 324         324 Survived      2
    ## 325         325     Died      3
    ## 326         326 Survived      1
    ## 327         327     Died      3
    ## 328         328 Survived      2
    ## 329         329 Survived      3
    ## 330         330 Survived      1
    ## 331         331 Survived      3
    ## 332         332     Died      1
    ## 333         333     Died      1
    ## 334         334     Died      3
    ## 335         335 Survived      1
    ## 336         336     Died      3
    ## 337         337     Died      1
    ## 338         338 Survived      1
    ## 339         339 Survived      3
    ## 340         340     Died      1
    ## 341         341 Survived      2
    ## 342         342 Survived      1
    ## 343         343     Died      2
    ## 344         344     Died      2
    ## 345         345     Died      2
    ## 346         346 Survived      2
    ## 347         347 Survived      2
    ## 348         348 Survived      3
    ## 349         349 Survived      3
    ## 350         350     Died      3
    ## 351         351     Died      3
    ## 352         352     Died      1
    ## 353         353     Died      3
    ## 354         354     Died      3
    ## 355         355     Died      3
    ## 356         356     Died      3
    ## 357         357 Survived      1
    ## 358         358     Died      2
    ## 359         359 Survived      3
    ## 360         360 Survived      3
    ## 361         361     Died      3
    ## 362         362     Died      2
    ## 363         363     Died      3
    ## 364         364     Died      3
    ## 365         365     Died      3
    ## 366         366     Died      3
    ## 367         367 Survived      1
    ## 368         368 Survived      3
    ## 369         369 Survived      3
    ## 370         370 Survived      1
    ## 371         371 Survived      1
    ## 372         372     Died      3
    ## 373         373     Died      3
    ## 374         374     Died      1
    ## 375         375     Died      3
    ## 376         376 Survived      1
    ## 377         377 Survived      3
    ## 378         378     Died      1
    ## 379         379     Died      3
    ## 380         380     Died      3
    ## 381         381 Survived      1
    ## 382         382 Survived      3
    ## 383         383     Died      3
    ## 384         384 Survived      1
    ## 385         385     Died      3
    ## 386         386     Died      2
    ## 387         387     Died      3
    ## 388         388 Survived      2
    ## 389         389     Died      3
    ## 390         390 Survived      2
    ## 391         391 Survived      1
    ## 392         392 Survived      3
    ## 393         393     Died      3
    ## 394         394 Survived      1
    ## 395         395 Survived      3
    ## 396         396     Died      3
    ## 397         397     Died      3
    ## 398         398     Died      2
    ## 399         399     Died      2
    ## 400         400 Survived      2
    ## 401         401 Survived      3
    ## 402         402     Died      3
    ## 403         403     Died      3
    ## 404         404     Died      3
    ## 405         405     Died      3
    ## 406         406     Died      2
    ## 407         407     Died      3
    ## 408         408 Survived      2
    ## 409         409     Died      3
    ## 410         410     Died      3
    ## 411         411     Died      3
    ## 412         412     Died      3
    ## 413         413 Survived      1
    ## 414         414     Died      2
    ## 415         415 Survived      3
    ## 416         416     Died      3
    ## 417         417 Survived      2
    ## 418         418 Survived      2
    ## 419         419     Died      2
    ## 420         420     Died      3
    ## 421         421     Died      3
    ## 422         422     Died      3
    ## 423         423     Died      3
    ## 424         424     Died      3
    ## 425         425     Died      3
    ## 426         426     Died      3
    ## 427         427 Survived      2
    ## 428         428 Survived      2
    ## 429         429     Died      3
    ## 430         430 Survived      3
    ## 431         431 Survived      1
    ## 432         432 Survived      3
    ## 433         433 Survived      2
    ## 434         434     Died      3
    ## 435         435     Died      1
    ## 436         436 Survived      1
    ## 437         437     Died      3
    ## 438         438 Survived      2
    ## 439         439     Died      1
    ## 440         440     Died      2
    ## 441         441 Survived      2
    ## 442         442     Died      3
    ## 443         443     Died      3
    ## 444         444 Survived      2
    ## 445         445 Survived      3
    ## 446         446 Survived      1
    ## 447         447 Survived      2
    ## 448         448 Survived      1
    ## 449         449 Survived      3
    ## 450         450 Survived      1
    ## 451         451     Died      2
    ## 452         452     Died      3
    ## 453         453     Died      1
    ## 454         454 Survived      1
    ## 455         455     Died      3
    ## 456         456 Survived      3
    ## 457         457     Died      1
    ## 458         458 Survived      1
    ## 459         459 Survived      2
    ## 460         460     Died      3
    ## 461         461 Survived      1
    ## 462         462     Died      3
    ## 463         463     Died      1
    ## 464         464     Died      2
    ## 465         465     Died      3
    ## 466         466     Died      3
    ## 467         467     Died      2
    ## 468         468     Died      1
    ## 469         469     Died      3
    ## 470         470 Survived      3
    ## 471         471     Died      3
    ## 472         472     Died      3
    ## 473         473 Survived      2
    ## 474         474 Survived      2
    ## 475         475     Died      3
    ## 476         476     Died      1
    ## 477         477     Died      2
    ## 478         478     Died      3
    ## 479         479     Died      3
    ## 480         480 Survived      3
    ## 481         481     Died      3
    ## 482         482     Died      2
    ## 483         483     Died      3
    ## 484         484 Survived      3
    ## 485         485 Survived      1
    ## 486         486     Died      3
    ## 487         487 Survived      1
    ## 488         488     Died      1
    ## 489         489     Died      3
    ## 490         490 Survived      3
    ## 491         491     Died      3
    ## 492         492     Died      3
    ## 493         493     Died      1
    ## 494         494     Died      1
    ## 495         495     Died      3
    ## 496         496     Died      3
    ## 497         497 Survived      1
    ## 498         498     Died      3
    ## 499         499     Died      1
    ## 500         500     Died      3
    ## 501         501     Died      3
    ## 502         502     Died      3
    ## 503         503     Died      3
    ## 504         504     Died      3
    ## 505         505 Survived      1
    ## 506         506     Died      1
    ## 507         507 Survived      2
    ## 508         508 Survived      1
    ## 509         509     Died      3
    ## 510         510 Survived      3
    ## 511         511 Survived      3
    ## 512         512     Died      3
    ## 513         513 Survived      1
    ## 514         514 Survived      1
    ## 515         515     Died      3
    ## 516         516     Died      1
    ## 517         517 Survived      2
    ## 518         518     Died      3
    ## 519         519 Survived      2
    ## 520         520     Died      3
    ## 521         521 Survived      1
    ## 522         522     Died      3
    ## 523         523     Died      3
    ## 524         524 Survived      1
    ## 525         525     Died      3
    ## 526         526     Died      3
    ## 527         527 Survived      2
    ## 528         528     Died      1
    ## 529         529     Died      3
    ## 530         530     Died      2
    ## 531         531 Survived      2
    ## 532         532     Died      3
    ## 533         533     Died      3
    ## 534         534 Survived      3
    ## 535         535     Died      3
    ## 536         536 Survived      2
    ## 537         537     Died      1
    ## 538         538 Survived      1
    ## 539         539     Died      3
    ## 540         540 Survived      1
    ## 541         541 Survived      1
    ## 542         542     Died      3
    ## 543         543     Died      3
    ## 544         544 Survived      2
    ## 545         545     Died      1
    ## 546         546     Died      1
    ## 547         547 Survived      2
    ## 548         548 Survived      2
    ## 549         549     Died      3
    ## 550         550 Survived      2
    ## 551         551 Survived      1
    ## 552         552     Died      2
    ## 553         553     Died      3
    ## 554         554 Survived      3
    ## 555         555 Survived      3
    ## 556         556     Died      1
    ## 557         557 Survived      1
    ## 558         558     Died      1
    ## 559         559 Survived      1
    ## 560         560 Survived      3
    ## 561         561     Died      3
    ## 562         562     Died      3
    ## 563         563     Died      2
    ## 564         564     Died      3
    ## 565         565     Died      3
    ## 566         566     Died      3
    ## 567         567     Died      3
    ## 568         568     Died      3
    ## 569         569     Died      3
    ## 570         570 Survived      3
    ## 571         571 Survived      2
    ## 572         572 Survived      1
    ## 573         573 Survived      1
    ## 574         574 Survived      3
    ## 575         575     Died      3
    ## 576         576     Died      3
    ## 577         577 Survived      2
    ## 578         578 Survived      1
    ## 579         579     Died      3
    ## 580         580 Survived      3
    ## 581         581 Survived      2
    ## 582         582 Survived      1
    ## 583         583     Died      2
    ## 584         584     Died      1
    ## 585         585     Died      3
    ## 586         586 Survived      1
    ## 587         587     Died      2
    ## 588         588 Survived      1
    ## 589         589     Died      3
    ## 590         590     Died      3
    ## 591         591     Died      3
    ## 592         592 Survived      1
    ## 593         593     Died      3
    ## 594         594     Died      3
    ## 595         595     Died      2
    ## 596         596     Died      3
    ## 597         597 Survived      2
    ## 598         598     Died      3
    ## 599         599     Died      3
    ## 600         600 Survived      1
    ## 601         601 Survived      2
    ## 602         602     Died      3
    ## 603         603     Died      1
    ## 604         604     Died      3
    ## 605         605 Survived      1
    ## 606         606     Died      3
    ## 607         607     Died      3
    ## 608         608 Survived      1
    ## 609         609 Survived      2
    ## 610         610 Survived      1
    ## 611         611     Died      3
    ## 612         612     Died      3
    ## 613         613 Survived      3
    ## 614         614     Died      3
    ## 615         615     Died      3
    ## 616         616 Survived      2
    ## 617         617     Died      3
    ## 618         618     Died      3
    ## 619         619 Survived      2
    ## 620         620     Died      2
    ## 621         621     Died      3
    ## 622         622 Survived      1
    ## 623         623 Survived      3
    ## 624         624     Died      3
    ## 625         625     Died      3
    ## 626         626     Died      1
    ## 627         627     Died      2
    ## 628         628 Survived      1
    ## 629         629     Died      3
    ## 630         630     Died      3
    ## 631         631 Survived      1
    ## 632         632     Died      3
    ## 633         633 Survived      1
    ## 634         634     Died      1
    ## 635         635     Died      3
    ## 636         636 Survived      2
    ## 637         637     Died      3
    ## 638         638     Died      2
    ## 639         639     Died      3
    ## 640         640     Died      3
    ## 641         641     Died      3
    ## 642         642 Survived      1
    ## 643         643     Died      3
    ## 644         644 Survived      3
    ## 645         645 Survived      3
    ## 646         646 Survived      1
    ## 647         647     Died      3
    ## 648         648 Survived      1
    ## 649         649     Died      3
    ## 650         650 Survived      3
    ## 651         651     Died      3
    ## 652         652 Survived      2
    ## 653         653     Died      3
    ## 654         654 Survived      3
    ## 655         655     Died      3
    ## 656         656     Died      2
    ## 657         657     Died      3
    ## 658         658     Died      3
    ## 659         659     Died      2
    ## 660         660     Died      1
    ## 661         661 Survived      1
    ## 662         662     Died      3
    ## 663         663     Died      1
    ## 664         664     Died      3
    ## 665         665 Survived      3
    ## 666         666     Died      2
    ## 667         667     Died      2
    ## 668         668     Died      3
    ## 669         669     Died      3
    ## 670         670 Survived      1
    ## 671         671 Survived      2
    ## 672         672     Died      1
    ## 673         673     Died      2
    ## 674         674 Survived      2
    ## 675         675     Died      2
    ## 676         676     Died      3
    ## 677         677     Died      3
    ## 678         678 Survived      3
    ## 679         679     Died      3
    ## 680         680 Survived      1
    ## 681         681     Died      3
    ## 682         682 Survived      1
    ## 683         683     Died      3
    ## 684         684     Died      3
    ## 685         685     Died      2
    ## 686         686     Died      2
    ## 687         687     Died      3
    ## 688         688     Died      3
    ## 689         689     Died      3
    ## 690         690 Survived      1
    ## 691         691 Survived      1
    ## 692         692 Survived      3
    ## 693         693 Survived      3
    ## 694         694     Died      3
    ## 695         695     Died      1
    ## 696         696     Died      2
    ## 697         697     Died      3
    ## 698         698 Survived      3
    ## 699         699     Died      1
    ## 700         700     Died      3
    ## 701         701 Survived      1
    ## 702         702 Survived      1
    ## 703         703     Died      3
    ## 704         704     Died      3
    ## 705         705     Died      3
    ## 706         706     Died      2
    ## 707         707 Survived      2
    ## 708         708 Survived      1
    ## 709         709 Survived      1
    ## 710         710 Survived      3
    ## 711         711 Survived      1
    ## 712         712     Died      1
    ## 713         713 Survived      1
    ## 714         714     Died      3
    ## 715         715     Died      2
    ## 716         716     Died      3
    ## 717         717 Survived      1
    ## 718         718 Survived      2
    ## 719         719     Died      3
    ## 720         720     Died      3
    ## 721         721 Survived      2
    ## 722         722     Died      3
    ## 723         723     Died      2
    ## 724         724     Died      2
    ## 725         725 Survived      1
    ## 726         726     Died      3
    ## 727         727 Survived      2
    ## 728         728 Survived      3
    ## 729         729     Died      2
    ## 730         730     Died      3
    ## 731         731 Survived      1
    ## 732         732     Died      3
    ## 733         733     Died      2
    ## 734         734     Died      2
    ## 735         735     Died      2
    ## 736         736     Died      3
    ## 737         737     Died      3
    ## 738         738 Survived      1
    ## 739         739     Died      3
    ## 740         740     Died      3
    ## 741         741 Survived      1
    ## 742         742     Died      1
    ## 743         743 Survived      1
    ## 744         744     Died      3
    ## 745         745 Survived      3
    ## 746         746     Died      1
    ## 747         747     Died      3
    ## 748         748 Survived      2
    ## 749         749     Died      1
    ## 750         750     Died      3
    ## 751         751 Survived      2
    ## 752         752 Survived      3
    ## 753         753     Died      3
    ## 754         754     Died      3
    ## 755         755 Survived      2
    ## 756         756 Survived      2
    ## 757         757     Died      3
    ## 758         758     Died      2
    ## 759         759     Died      3
    ## 760         760 Survived      1
    ## 761         761     Died      3
    ## 762         762     Died      3
    ## 763         763 Survived      3
    ## 764         764 Survived      1
    ## 765         765     Died      3
    ## 766         766 Survived      1
    ## 767         767     Died      1
    ## 768         768     Died      3
    ## 769         769     Died      3
    ## 770         770     Died      3
    ## 771         771     Died      3
    ## 772         772     Died      3
    ## 773         773     Died      2
    ## 774         774     Died      3
    ## 775         775 Survived      2
    ## 776         776     Died      3
    ## 777         777     Died      3
    ## 778         778 Survived      3
    ## 779         779     Died      3
    ## 780         780 Survived      1
    ## 781         781 Survived      3
    ## 782         782 Survived      1
    ## 783         783     Died      1
    ## 784         784     Died      3
    ## 785         785     Died      3
    ## 786         786     Died      3
    ## 787         787 Survived      3
    ## 788         788     Died      3
    ## 789         789 Survived      3
    ## 790         790     Died      1
    ## 791         791     Died      3
    ## 792         792     Died      2
    ## 793         793     Died      3
    ## 794         794     Died      1
    ## 795         795     Died      3
    ## 796         796     Died      2
    ## 797         797 Survived      1
    ## 798         798 Survived      3
    ## 799         799     Died      3
    ## 800         800     Died      3
    ## 801         801     Died      2
    ## 802         802 Survived      2
    ## 803         803 Survived      1
    ## 804         804 Survived      3
    ## 805         805 Survived      3
    ## 806         806     Died      3
    ## 807         807     Died      1
    ## 808         808     Died      3
    ## 809         809     Died      2
    ## 810         810 Survived      1
    ## 811         811     Died      3
    ## 812         812     Died      3
    ## 813         813     Died      2
    ## 814         814     Died      3
    ## 815         815     Died      3
    ## 816         816     Died      1
    ## 817         817     Died      3
    ## 818         818     Died      2
    ## 819         819     Died      3
    ## 820         820     Died      3
    ## 821         821 Survived      1
    ## 822         822 Survived      3
    ## 823         823     Died      1
    ## 824         824 Survived      3
    ## 825         825     Died      3
    ## 826         826     Died      3
    ## 827         827     Died      3
    ## 828         828 Survived      2
    ## 829         829 Survived      3
    ## 830         830 Survived      1
    ## 831         831 Survived      3
    ## 832         832 Survived      2
    ## 833         833     Died      3
    ## 834         834     Died      3
    ## 835         835     Died      3
    ## 836         836 Survived      1
    ## 837         837     Died      3
    ## 838         838     Died      3
    ## 839         839 Survived      3
    ## 840         840 Survived      1
    ## 841         841     Died      3
    ## 842         842     Died      2
    ## 843         843 Survived      1
    ## 844         844     Died      3
    ## 845         845     Died      3
    ## 846         846     Died      3
    ## 847         847     Died      3
    ## 848         848     Died      3
    ## 849         849     Died      2
    ## 850         850 Survived      1
    ## 851         851     Died      3
    ## 852         852     Died      3
    ## 853         853     Died      3
    ## 854         854 Survived      1
    ## 855         855     Died      2
    ## 856         856 Survived      3
    ## 857         857 Survived      1
    ## 858         858 Survived      1
    ## 859         859 Survived      3
    ## 860         860     Died      3
    ## 861         861     Died      3
    ## 862         862     Died      2
    ## 863         863 Survived      1
    ## 864         864     Died      3
    ## 865         865     Died      2
    ## 866         866 Survived      2
    ## 867         867 Survived      2
    ## 868         868     Died      1
    ## 869         869     Died      3
    ## 870         870 Survived      3
    ## 871         871     Died      3
    ## 872         872 Survived      1
    ## 873         873     Died      1
    ## 874         874     Died      3
    ## 875         875 Survived      2
    ## 876         876 Survived      3
    ## 877         877     Died      3
    ## 878         878     Died      3
    ## 879         879     Died      3
    ## 880         880 Survived      1
    ## 881         881 Survived      2
    ## 882         882     Died      3
    ## 883         883     Died      3
    ## 884         884     Died      2
    ## 885         885     Died      3
    ## 886         886     Died      3
    ## 887         887     Died      2
    ## 888         888 Survived      1
    ## 889         889     Died      3
    ## 890         890 Survived      1
    ## 891         891     Died      3
    ##                                                                                   Name
    ## 1                                                              Braund, Mr. Owen Harris
    ## 2                                  Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    ## 3                                                               Heikkinen, Miss. Laina
    ## 4                                         Futrelle, Mrs. Jacques Heath (Lily May Peel)
    ## 5                                                             Allen, Mr. William Henry
    ## 6                                                                     Moran, Mr. James
    ## 7                                                              McCarthy, Mr. Timothy J
    ## 8                                                       Palsson, Master. Gosta Leonard
    ## 9                                    Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)
    ## 10                                                 Nasser, Mrs. Nicholas (Adele Achem)
    ## 11                                                     Sandstrom, Miss. Marguerite Rut
    ## 12                                                            Bonnell, Miss. Elizabeth
    ## 13                                                      Saundercock, Mr. William Henry
    ## 14                                                         Andersson, Mr. Anders Johan
    ## 15                                                Vestrom, Miss. Hulda Amanda Adolfina
    ## 16                                                    Hewlett, Mrs. (Mary D Kingcome) 
    ## 17                                                                Rice, Master. Eugene
    ## 18                                                        Williams, Mr. Charles Eugene
    ## 19                             Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)
    ## 20                                                             Masselmani, Mrs. Fatima
    ## 21                                                                Fynney, Mr. Joseph J
    ## 22                                                               Beesley, Mr. Lawrence
    ## 23                                                         McGowan, Miss. Anna "Annie"
    ## 24                                                        Sloper, Mr. William Thompson
    ## 25                                                       Palsson, Miss. Torborg Danira
    ## 26                           Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson)
    ## 27                                                             Emir, Mr. Farred Chehab
    ## 28                                                      Fortune, Mr. Charles Alexander
    ## 29                                                       O'Dwyer, Miss. Ellen "Nellie"
    ## 30                                                                 Todoroff, Mr. Lalio
    ## 31                                                            Uruchurtu, Don. Manuel E
    ## 32                                      Spencer, Mrs. William Augustus (Marie Eugenie)
    ## 33                                                            Glynn, Miss. Mary Agatha
    ## 34                                                               Wheadon, Mr. Edward H
    ## 35                                                             Meyer, Mr. Edgar Joseph
    ## 36                                                      Holverson, Mr. Alexander Oskar
    ## 37                                                                    Mamee, Mr. Hanna
    ## 38                                                            Cann, Mr. Ernest Charles
    ## 39                                                  Vander Planke, Miss. Augusta Maria
    ## 40                                                         Nicola-Yarred, Miss. Jamila
    ## 41                                      Ahlin, Mrs. Johan (Johanna Persdotter Larsson)
    ## 42                            Turpin, Mrs. William John Robert (Dorothy Ann Wonnacott)
    ## 43                                                                 Kraeff, Mr. Theodor
    ## 44                                            Laroche, Miss. Simonne Marie Anne Andree
    ## 45                                                       Devaney, Miss. Margaret Delia
    ## 46                                                            Rogers, Mr. William John
    ## 47                                                                   Lennon, Mr. Denis
    ## 48                                                           O'Driscoll, Miss. Bridget
    ## 49                                                                 Samaan, Mr. Youssef
    ## 50                                       Arnold-Franchi, Mrs. Josef (Josefine Franchi)
    ## 51                                                          Panula, Master. Juha Niilo
    ## 52                                                        Nosworthy, Mr. Richard Cater
    ## 53                                            Harper, Mrs. Henry Sleeper (Myna Haxtun)
    ## 54                                  Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkinson)
    ## 55                                                      Ostby, Mr. Engelhart Cornelius
    ## 56                                                                   Woolner, Mr. Hugh
    ## 57                                                                   Rugg, Miss. Emily
    ## 58                                                                 Novel, Mr. Mansouer
    ## 59                                                        West, Miss. Constance Mirium
    ## 60                                                  Goodwin, Master. William Frederick
    ## 61                                                               Sirayanian, Mr. Orsen
    ## 62                                                                 Icard, Miss. Amelie
    ## 63                                                         Harris, Mr. Henry Birkhardt
    ## 64                                                               Skoog, Master. Harald
    ## 65                                                               Stewart, Mr. Albert A
    ## 66                                                            Moubarek, Master. Gerios
    ## 67                                                        Nye, Mrs. (Elizabeth Ramell)
    ## 68                                                            Crease, Mr. Ernest James
    ## 69                                                     Andersson, Miss. Erna Alexandra
    ## 70                                                                   Kink, Mr. Vincenz
    ## 71                                                          Jenkin, Mr. Stephen Curnow
    ## 72                                                          Goodwin, Miss. Lillian Amy
    ## 73                                                                Hood, Mr. Ambrose Jr
    ## 74                                                         Chronopoulos, Mr. Apostolos
    ## 75                                                                       Bing, Mr. Lee
    ## 76                                                             Moen, Mr. Sigurd Hansen
    ## 77                                                                   Staneff, Mr. Ivan
    ## 78                                                            Moutal, Mr. Rahamin Haim
    ## 79                                                       Caldwell, Master. Alden Gates
    ## 80                                                            Dowdell, Miss. Elizabeth
    ## 81                                                                Waelens, Mr. Achille
    ## 82                                                         Sheerlinck, Mr. Jan Baptist
    ## 83                                                      McDermott, Miss. Brigdet Delia
    ## 84                                                             Carrau, Mr. Francisco M
    ## 85                                                                 Ilett, Miss. Bertha
    ## 86                             Backstrom, Mrs. Karl Alfred (Maria Mathilda Gustafsson)
    ## 87                                                              Ford, Mr. William Neal
    ## 88                                                       Slocovski, Mr. Selman Francis
    ## 89                                                          Fortune, Miss. Mabel Helen
    ## 90                                                              Celotti, Mr. Francesco
    ## 91                                                                Christmann, Mr. Emil
    ## 92                                                          Andreasson, Mr. Paul Edvin
    ## 93                                                         Chaffee, Mr. Herbert Fuller
    ## 94                                                             Dean, Mr. Bertram Frank
    ## 95                                                                   Coxon, Mr. Daniel
    ## 96                                                         Shorney, Mr. Charles Joseph
    ## 97                                                           Goldschmidt, Mr. George B
    ## 98                                                     Greenfield, Mr. William Bertram
    ## 99                                                Doling, Mrs. John T (Ada Julia Bone)
    ## 100                                                                  Kantor, Mr. Sinai
    ## 101                                                            Petranec, Miss. Matilda
    ## 102                                                   Petroff, Mr. Pastcho ("Pentcho")
    ## 103                                                          White, Mr. Richard Frasar
    ## 104                                                         Johansson, Mr. Gustaf Joel
    ## 105                                                     Gustafsson, Mr. Anders Vilhelm
    ## 106                                                              Mionoff, Mr. Stoytcho
    ## 107                                                   Salkjelsvik, Miss. Anna Kristine
    ## 108                                                             Moss, Mr. Albert Johan
    ## 109                                                                    Rekic, Mr. Tido
    ## 110                                                                Moran, Miss. Bertha
    ## 111                                                     Porter, Mr. Walter Chamberlain
    ## 112                                                               Zabour, Miss. Hileni
    ## 113                                                             Barton, Mr. David John
    ## 114                                                            Jussila, Miss. Katriina
    ## 115                                                              Attalah, Miss. Malake
    ## 116                                                              Pekoniemi, Mr. Edvard
    ## 117                                                               Connors, Mr. Patrick
    ## 118                                                    Turpin, Mr. William John Robert
    ## 119                                                           Baxter, Mr. Quigg Edmond
    ## 120                                                  Andersson, Miss. Ellis Anna Maria
    ## 121                                                        Hickman, Mr. Stanley George
    ## 122                                                         Moore, Mr. Leonard Charles
    ## 123                                                               Nasser, Mr. Nicholas
    ## 124                                                                Webber, Miss. Susan
    ## 125                                                        White, Mr. Percival Wayland
    ## 126                                                       Nicola-Yarred, Master. Elias
    ## 127                                                                McMahon, Mr. Martin
    ## 128                                                          Madsen, Mr. Fridtjof Arne
    ## 129                                                                  Peter, Miss. Anna
    ## 130                                                                 Ekstrom, Mr. Johan
    ## 131                                                               Drazenoic, Mr. Jozef
    ## 132                                                     Coelho, Mr. Domingos Fernandeo
    ## 133                                     Robins, Mrs. Alexander A (Grace Charity Laury)
    ## 134                                      Weisz, Mrs. Leopold (Mathilde Francoise Pede)
    ## 135                                                     Sobey, Mr. Samuel James Hayden
    ## 136                                                                 Richard, Mr. Emile
    ## 137                                                       Newsom, Miss. Helen Monypeny
    ## 138                                                        Futrelle, Mr. Jacques Heath
    ## 139                                                                Osen, Mr. Olaf Elon
    ## 140                                                                 Giglio, Mr. Victor
    ## 141                                                      Boulos, Mrs. Joseph (Sultana)
    ## 142                                                           Nysten, Miss. Anna Sofia
    ## 143                               Hakkarainen, Mrs. Pekka Pietari (Elin Matilda Dolck)
    ## 144                                                                Burke, Mr. Jeremiah
    ## 145                                                         Andrew, Mr. Edgardo Samuel
    ## 146                                                       Nicholls, Mr. Joseph Charles
    ## 147                                       Andersson, Mr. August Edvard ("Wennerstrom")
    ## 148                                                   Ford, Miss. Robina Maggie "Ruby"
    ## 149                                           Navratil, Mr. Michel ("Louis M Hoffman")
    ## 150                                                  Byles, Rev. Thomas Roussel Davids
    ## 151                                                         Bateman, Rev. Robert James
    ## 152                                                  Pears, Mrs. Thomas (Edith Wearne)
    ## 153                                                                   Meo, Mr. Alfonzo
    ## 154                                                    van Billiard, Mr. Austin Blyler
    ## 155                                                              Olsen, Mr. Ole Martin
    ## 156                                                        Williams, Mr. Charles Duane
    ## 157                                                   Gilnagh, Miss. Katherine "Katie"
    ## 158                                                                    Corn, Mr. Harry
    ## 159                                                                Smiljanic, Mr. Mile
    ## 160                                                         Sage, Master. Thomas Henry
    ## 161                                                           Cribb, Mr. John Hatfield
    ## 162                                 Watt, Mrs. James (Elizabeth "Bessie" Inglis Milne)
    ## 163                                                         Bengtsson, Mr. John Viktor
    ## 164                                                                    Calic, Mr. Jovo
    ## 165                                                       Panula, Master. Eino Viljami
    ## 166                                    Goldsmith, Master. Frank John William "Frankie"
    ## 167                                             Chibnall, Mrs. (Edith Martha Bowerman)
    ## 168                                    Skoog, Mrs. William (Anna Bernhardina Karlsson)
    ## 169                                                                Baumann, Mr. John D
    ## 170                                                                      Ling, Mr. Lee
    ## 171                                                          Van der hoef, Mr. Wyckoff
    ## 172                                                               Rice, Master. Arthur
    ## 173                                                       Johnson, Miss. Eleanor Ileen
    ## 174                                                          Sivola, Mr. Antti Wilhelm
    ## 175                                                            Smith, Mr. James Clinch
    ## 176                                                             Klasen, Mr. Klas Albin
    ## 177                                                      Lefebre, Master. Henry Forbes
    ## 178                                                         Isham, Miss. Ann Elizabeth
    ## 179                                                                 Hale, Mr. Reginald
    ## 180                                                                Leonard, Mr. Lionel
    ## 181                                                       Sage, Miss. Constance Gladys
    ## 182                                                                   Pernot, Mr. Rene
    ## 183                                              Asplund, Master. Clarence Gustaf Hugo
    ## 184                                                          Becker, Master. Richard F
    ## 185                                                Kink-Heilmann, Miss. Luise Gretchen
    ## 186                                                              Rood, Mr. Hugh Roscoe
    ## 187                                    O'Brien, Mrs. Thomas (Johanna "Hannah" Godfrey)
    ## 188                                      Romaine, Mr. Charles Hallace ("Mr C Rolmane")
    ## 189                                                                   Bourke, Mr. John
    ## 190                                                                Turcin, Mr. Stjepan
    ## 191                                                                Pinsky, Mrs. (Rosa)
    ## 192                                                              Carbines, Mr. William
    ## 193                                    Andersen-Jensen, Miss. Carla Christine Nielsine
    ## 194                                                         Navratil, Master. Michel M
    ## 195                                          Brown, Mrs. James Joseph (Margaret Tobin)
    ## 196                                                               Lurette, Miss. Elise
    ## 197                                                                Mernagh, Mr. Robert
    ## 198                                                   Olsen, Mr. Karl Siegwart Andreas
    ## 199                                                   Madigan, Miss. Margaret "Maggie"
    ## 200                                             Yrois, Miss. Henriette ("Mrs Harbeck")
    ## 201                                                     Vande Walle, Mr. Nestor Cyriel
    ## 202                                                                Sage, Mr. Frederick
    ## 203                                                         Johanson, Mr. Jakob Alfred
    ## 204                                                               Youseff, Mr. Gerious
    ## 205                                                           Cohen, Mr. Gurshon "Gus"
    ## 206                                                         Strom, Miss. Telma Matilda
    ## 207                                                         Backstrom, Mr. Karl Alfred
    ## 208                                                        Albimona, Mr. Nassef Cassem
    ## 209                                                          Carr, Miss. Helen "Ellen"
    ## 210                                                                   Blank, Mr. Henry
    ## 211                                                                     Ali, Mr. Ahmed
    ## 212                                                         Cameron, Miss. Clear Annie
    ## 213                                                             Perkin, Mr. John Henry
    ## 214                                                        Givard, Mr. Hans Kristensen
    ## 215                                                                Kiernan, Mr. Philip
    ## 216                                                            Newell, Miss. Madeleine
    ## 217                                                             Honkanen, Miss. Eliina
    ## 218                                                       Jacobsohn, Mr. Sidney Samuel
    ## 219                                                              Bazzani, Miss. Albina
    ## 220                                                                 Harris, Mr. Walter
    ## 221                                                     Sunderland, Mr. Victor Francis
    ## 222                                                               Bracken, Mr. James H
    ## 223                                                            Green, Mr. George Henry
    ## 224                                                               Nenkoff, Mr. Christo
    ## 225                                                       Hoyt, Mr. Frederick Maxfield
    ## 226                                                       Berglund, Mr. Karl Ivar Sven
    ## 227                                                          Mellors, Mr. William John
    ## 228                                                    Lovell, Mr. John Hall ("Henry")
    ## 229                                                          Fahlstrom, Mr. Arne Jonas
    ## 230                                                            Lefebre, Miss. Mathilde
    ## 231                                       Harris, Mrs. Henry Birkhardt (Irene Wallach)
    ## 232                                                           Larsson, Mr. Bengt Edvin
    ## 233                                                          Sjostedt, Mr. Ernst Adolf
    ## 234                                                     Asplund, Miss. Lillian Gertrud
    ## 235                                                  Leyson, Mr. Robert William Norman
    ## 236                                                       Harknett, Miss. Alice Phoebe
    ## 237                                                                  Hold, Mr. Stephen
    ## 238                                                   Collyer, Miss. Marjorie "Lottie"
    ## 239                                                    Pengelly, Mr. Frederick William
    ## 240                                                             Hunt, Mr. George Henry
    ## 241                                                              Zabour, Miss. Thamine
    ## 242                                                     Murphy, Miss. Katherine "Kate"
    ## 243                                                    Coleridge, Mr. Reginald Charles
    ## 244                                                      Maenpaa, Mr. Matti Alexanteri
    ## 245                                                               Attalah, Mr. Sleiman
    ## 246                                                        Minahan, Dr. William Edward
    ## 247                                              Lindahl, Miss. Agda Thorilda Viktoria
    ## 248                                                    Hamalainen, Mrs. William (Anna)
    ## 249                                                      Beckwith, Mr. Richard Leonard
    ## 250                                                      Carter, Rev. Ernest Courtenay
    ## 251                                                             Reed, Mr. James George
    ## 252                                         Strom, Mrs. Wilhelm (Elna Matilda Persson)
    ## 253                                                          Stead, Mr. William Thomas
    ## 254                                                           Lobb, Mr. William Arthur
    ## 255                                           Rosblom, Mrs. Viktor (Helena Wilhelmina)
    ## 256                                            Touma, Mrs. Darwis (Hanne Youssef Razi)
    ## 257                                                     Thorne, Mrs. Gertrude Maybelle
    ## 258                                                               Cherry, Miss. Gladys
    ## 259                                                                   Ward, Miss. Anna
    ## 260                                                        Parrish, Mrs. (Lutie Davis)
    ## 261                                                                  Smith, Mr. Thomas
    ## 262                                                  Asplund, Master. Edvin Rojj Felix
    ## 263                                                                  Taussig, Mr. Emil
    ## 264                                                              Harrison, Mr. William
    ## 265                                                                 Henry, Miss. Delia
    ## 266                                                                  Reeves, Mr. David
    ## 267                                                          Panula, Mr. Ernesti Arvid
    ## 268                                                           Persson, Mr. Ernst Ulrik
    ## 269                                      Graham, Mrs. William Thompson (Edith Junkins)
    ## 270                                                             Bissette, Miss. Amelia
    ## 271                                                              Cairns, Mr. Alexander
    ## 272                                                       Tornquist, Mr. William Henry
    ## 273                                          Mellinger, Mrs. (Elizabeth Anne Maidment)
    ## 274                                                              Natsch, Mr. Charles H
    ## 275                                                         Healy, Miss. Hanora "Nora"
    ## 276                                                  Andrews, Miss. Kornelia Theodosia
    ## 277                                                  Lindblom, Miss. Augusta Charlotta
    ## 278                                                        Parkes, Mr. Francis "Frank"
    ## 279                                                                 Rice, Master. Eric
    ## 280                                                   Abbott, Mrs. Stanton (Rosa Hunt)
    ## 281                                                                   Duane, Mr. Frank
    ## 282                                                   Olsson, Mr. Nils Johan Goransson
    ## 283                                                          de Pelsmaeker, Mr. Alfons
    ## 284                                                         Dorking, Mr. Edward Arthur
    ## 285                                                         Smith, Mr. Richard William
    ## 286                                                                Stankovic, Mr. Ivan
    ## 287                                                            de Mulder, Mr. Theodore
    ## 288                                                               Naidenoff, Mr. Penko
    ## 289                                                               Hosono, Mr. Masabumi
    ## 290                                                               Connolly, Miss. Kate
    ## 291                                                       Barber, Miss. Ellen "Nellie"
    ## 292                                            Bishop, Mrs. Dickinson H (Helen Walton)
    ## 293                                                             Levy, Mr. Rene Jacques
    ## 294                                                                Haas, Miss. Aloisia
    ## 295                                                                   Mineff, Mr. Ivan
    ## 296                                                                  Lewy, Mr. Ervin G
    ## 297                                                                 Hanna, Mr. Mansour
    ## 298                                                       Allison, Miss. Helen Loraine
    ## 299                                                              Saalfeld, Mr. Adolphe
    ## 300                                    Baxter, Mrs. James (Helene DeLaudeniere Chaput)
    ## 301                                           Kelly, Miss. Anna Katherine "Annie Kate"
    ## 302                                                                 McCoy, Mr. Bernard
    ## 303                                                    Johnson, Mr. William Cahoone Jr
    ## 304                                                                Keane, Miss. Nora A
    ## 305                                                  Williams, Mr. Howard Hugh "Harry"
    ## 306                                                     Allison, Master. Hudson Trevor
    ## 307                                                            Fleming, Miss. Margaret
    ## 308 Penasco y Castellana, Mrs. Victor de Satode (Maria Josefa Perez de Soto y Vallejo)
    ## 309                                                                Abelson, Mr. Samuel
    ## 310                                                     Francatelli, Miss. Laura Mabel
    ## 311                                                     Hays, Miss. Margaret Bechstein
    ## 312                                                         Ryerson, Miss. Emily Borie
    ## 313                                              Lahtinen, Mrs. William (Anna Sylfven)
    ## 314                                                             Hendekovic, Mr. Ignjac
    ## 315                                                                 Hart, Mr. Benjamin
    ## 316                                                    Nilsson, Miss. Helmina Josefina
    ## 317                                                Kantor, Mrs. Sinai (Miriam Sternin)
    ## 318                                                               Moraweck, Dr. Ernest
    ## 319                                                           Wick, Miss. Mary Natalie
    ## 320                           Spedden, Mrs. Frederic Oakley (Margaretta Corning Stone)
    ## 321                                                                 Dennis, Mr. Samuel
    ## 322                                                                   Danoff, Mr. Yoto
    ## 323                                                          Slayter, Miss. Hilda Mary
    ## 324                                Caldwell, Mrs. Albert Francis (Sylvia Mae Harbaugh)
    ## 325                                                           Sage, Mr. George John Jr
    ## 326                                                           Young, Miss. Marie Grice
    ## 327                                                          Nysveen, Mr. Johan Hansen
    ## 328                                                            Ball, Mrs. (Ada E Hall)
    ## 329                                     Goldsmith, Mrs. Frank John (Emily Alice Brown)
    ## 330                                                       Hippach, Miss. Jean Gertrude
    ## 331                                                                 McCoy, Miss. Agnes
    ## 332                                                                Partner, Mr. Austen
    ## 333                                                          Graham, Mr. George Edward
    ## 334                                                    Vander Planke, Mr. Leo Edmondus
    ## 335                                 Frauenthal, Mrs. Henry William (Clara Heinsheimer)
    ## 336                                                                 Denkoff, Mr. Mitto
    ## 337                                                          Pears, Mr. Thomas Clinton
    ## 338                                                    Burns, Miss. Elizabeth Margaret
    ## 339                                                              Dahl, Mr. Karl Edwart
    ## 340                                                       Blackwell, Mr. Stephen Weart
    ## 341                                                     Navratil, Master. Edmond Roger
    ## 342                                                     Fortune, Miss. Alice Elizabeth
    ## 343                                                         Collander, Mr. Erik Gustaf
    ## 344                                         Sedgwick, Mr. Charles Frederick Waddington
    ## 345                                                            Fox, Mr. Stanley Hubert
    ## 346                                                      Brown, Miss. Amelia "Mildred"
    ## 347                                                          Smith, Miss. Marion Elsie
    ## 348                                          Davison, Mrs. Thomas Henry (Mary E Finck)
    ## 349                                             Coutts, Master. William Loch "William"
    ## 350                                                                   Dimic, Mr. Jovan
    ## 351                                                             Odahl, Mr. Nils Martin
    ## 352                                             Williams-Lambert, Mr. Fletcher Fellows
    ## 353                                                                 Elias, Mr. Tannous
    ## 354                                                          Arnold-Franchi, Mr. Josef
    ## 355                                                                  Yousif, Mr. Wazli
    ## 356                                                        Vanden Steen, Mr. Leo Peter
    ## 357                                                        Bowerman, Miss. Elsie Edith
    ## 358                                                          Funk, Miss. Annie Clemmer
    ## 359                                                               McGovern, Miss. Mary
    ## 360                                                  Mockler, Miss. Helen Mary "Ellie"
    ## 361                                                                 Skoog, Mr. Wilhelm
    ## 362                                                          del Carlo, Mr. Sebastiano
    ## 363                                                    Barbara, Mrs. (Catherine David)
    ## 364                                                                    Asim, Mr. Adola
    ## 365                                                                O'Brien, Mr. Thomas
    ## 366                                                     Adahl, Mr. Mauritz Nils Martin
    ## 367                                   Warren, Mrs. Frank Manley (Anna Sophia Atkinson)
    ## 368                                                     Moussa, Mrs. (Mantoura Boulos)
    ## 369                                                                Jermyn, Miss. Annie
    ## 370                                                      Aubart, Mme. Leontine Pauline
    ## 371                                                        Harder, Mr. George Achilles
    ## 372                                                          Wiklund, Mr. Jakob Alfred
    ## 373                                                         Beavan, Mr. William Thomas
    ## 374                                                                Ringhini, Mr. Sante
    ## 375                                                         Palsson, Miss. Stina Viola
    ## 376                                              Meyer, Mrs. Edgar Joseph (Leila Saks)
    ## 377                                                    Landergren, Miss. Aurora Adelia
    ## 378                                                          Widener, Mr. Harry Elkins
    ## 379                                                                Betros, Mr. Tannous
    ## 380                                                        Gustafsson, Mr. Karl Gideon
    ## 381                                                              Bidois, Miss. Rosalie
    ## 382                                                        Nakid, Miss. Maria ("Mary")
    ## 383                                                                 Tikkanen, Mr. Juho
    ## 384                                Holverson, Mrs. Alexander Oskar (Mary Aline Towner)
    ## 385                                                             Plotcharsky, Mr. Vasil
    ## 386                                                          Davies, Mr. Charles Henry
    ## 387                                                    Goodwin, Master. Sidney Leonard
    ## 388                                                                   Buss, Miss. Kate
    ## 389                                                               Sadlier, Mr. Matthew
    ## 390                                                              Lehmann, Miss. Bertha
    ## 391                                                         Carter, Mr. William Ernest
    ## 392                                                             Jansson, Mr. Carl Olof
    ## 393                                                       Gustafsson, Mr. Johan Birger
    ## 394                                                             Newell, Miss. Marjorie
    ## 395                                Sandstrom, Mrs. Hjalmar (Agnes Charlotta Bengtsson)
    ## 396                                                                Johansson, Mr. Erik
    ## 397                                                                Olsson, Miss. Elina
    ## 398                                                            McKane, Mr. Peter David
    ## 399                                                                   Pain, Dr. Alfred
    ## 400                                                   Trout, Mrs. William H (Jessie L)
    ## 401                                                                 Niskanen, Mr. Juha
    ## 402                                                                    Adams, Mr. John
    ## 403                                                           Jussila, Miss. Mari Aina
    ## 404                                                     Hakkarainen, Mr. Pekka Pietari
    ## 405                                                            Oreskovic, Miss. Marija
    ## 406                                                                 Gale, Mr. Shadrach
    ## 407                                                   Widegren, Mr. Carl/Charles Peter
    ## 408                                                     Richards, Master. William Rowe
    ## 409                                                  Birkeland, Mr. Hans Martin Monsen
    ## 410                                                                 Lefebre, Miss. Ida
    ## 411                                                                 Sdycoff, Mr. Todor
    ## 412                                                                    Hart, Mr. Henry
    ## 413                                                             Minahan, Miss. Daisy E
    ## 414                                                     Cunningham, Mr. Alfred Fleming
    ## 415                                                          Sundman, Mr. Johan Julian
    ## 416                                            Meek, Mrs. Thomas (Annie Louise Rowley)
    ## 417                                    Drew, Mrs. James Vivian (Lulu Thorne Christian)
    ## 418                                                      Silven, Miss. Lyyli Karoliina
    ## 419                                                         Matthews, Mr. William John
    ## 420                                                          Van Impe, Miss. Catharina
    ## 421                                                             Gheorgheff, Mr. Stanio
    ## 422                                                                Charters, Mr. David
    ## 423                                                                 Zimmerman, Mr. Leo
    ## 424                             Danbom, Mrs. Ernst Gilbert (Anna Sigrid Maria Brogren)
    ## 425                                                        Rosblom, Mr. Viktor Richard
    ## 426                                                             Wiseman, Mr. Phillippe
    ## 427                                        Clarke, Mrs. Charles V (Ada Maria Winfield)
    ## 428                Phillips, Miss. Kate Florence ("Mrs Kate Louise Phillips Marshall")
    ## 429                                                                   Flynn, Mr. James
    ## 430                                                 Pickard, Mr. Berk (Berk Trembisky)
    ## 431                                          Bjornstrom-Steffansson, Mr. Mauritz Hakan
    ## 432                                  Thorneycroft, Mrs. Percival (Florence Kate White)
    ## 433                                Louch, Mrs. Charles Alexander (Alice Adelaide Slow)
    ## 434                                                         Kallio, Mr. Nikolai Erland
    ## 435                                                          Silvey, Mr. William Baird
    ## 436                                                          Carter, Miss. Lucile Polk
    ## 437                                               Ford, Miss. Doolina Margaret "Daisy"
    ## 438                                              Richards, Mrs. Sidney (Emily Hocking)
    ## 439                                                                  Fortune, Mr. Mark
    ## 440                                             Kvillner, Mr. Johan Henrik Johannesson
    ## 441                                        Hart, Mrs. Benjamin (Esther Ada Bloomfield)
    ## 442                                                                    Hampe, Mr. Leon
    ## 443                                                          Petterson, Mr. Johan Emil
    ## 444                                                          Reynaldo, Ms. Encarnacion
    ## 445                                                  Johannesen-Bratthammer, Mr. Bernt
    ## 446                                                          Dodge, Master. Washington
    ## 447                                                  Mellinger, Miss. Madeleine Violet
    ## 448                                                        Seward, Mr. Frederic Kimber
    ## 449                                                     Baclini, Miss. Marie Catherine
    ## 450                                                     Peuchen, Major. Arthur Godfrey
    ## 451                                                              West, Mr. Edwy Arthur
    ## 452                                                    Hagland, Mr. Ingvald Olai Olsen
    ## 453                                                    Foreman, Mr. Benjamin Laventall
    ## 454                                                           Goldenberg, Mr. Samuel L
    ## 455                                                                Peduzzi, Mr. Joseph
    ## 456                                                                 Jalsevac, Mr. Ivan
    ## 457                                                          Millet, Mr. Francis Davis
    ## 458                                                  Kenyon, Mrs. Frederick R (Marion)
    ## 459                                                                Toomey, Miss. Ellen
    ## 460                                                              O'Connor, Mr. Maurice
    ## 461                                                                Anderson, Mr. Harry
    ## 462                                                                Morley, Mr. William
    ## 463                                                                  Gee, Mr. Arthur H
    ## 464                                                       Milling, Mr. Jacob Christian
    ## 465                                                                 Maisner, Mr. Simon
    ## 466                                                    Goncalves, Mr. Manuel Estanslas
    ## 467                                                              Campbell, Mr. William
    ## 468                                                         Smart, Mr. John Montgomery
    ## 469                                                                 Scanlan, Mr. James
    ## 470                                                      Baclini, Miss. Helene Barbara
    ## 471                                                                  Keefe, Mr. Arthur
    ## 472                                                                    Cacic, Mr. Luka
    ## 473                                            West, Mrs. Edwy Arthur (Ada Mary Worth)
    ## 474                                       Jerwan, Mrs. Amin S (Marie Marthe Thuillard)
    ## 475                                                        Strandberg, Miss. Ida Sofia
    ## 476                                                        Clifford, Mr. George Quincy
    ## 477                                                            Renouf, Mr. Peter Henry
    ## 478                                                          Braund, Mr. Lewis Richard
    ## 479                                                          Karlsson, Mr. Nils August
    ## 480                                                           Hirvonen, Miss. Hildur E
    ## 481                                                     Goodwin, Master. Harold Victor
    ## 482                                                   Frost, Mr. Anthony Wood "Archie"
    ## 483                                                           Rouse, Mr. Richard Henry
    ## 484                                                             Turkula, Mrs. (Hedwig)
    ## 485                                                            Bishop, Mr. Dickinson H
    ## 486                                                             Lefebre, Miss. Jeannie
    ## 487                                    Hoyt, Mrs. Frederick Maxfield (Jane Anne Forby)
    ## 488                                                            Kent, Mr. Edward Austin
    ## 489                                                      Somerton, Mr. Francis William
    ## 490                                              Coutts, Master. Eden Leslie "Neville"
    ## 491                                               Hagland, Mr. Konrad Mathias Reiersen
    ## 492                                                                Windelov, Mr. Einar
    ## 493                                                         Molson, Mr. Harry Markland
    ## 494                                                            Artagaveytia, Mr. Ramon
    ## 495                                                         Stanley, Mr. Edward Roland
    ## 496                                                              Yousseff, Mr. Gerious
    ## 497                                                     Eustis, Miss. Elizabeth Mussey
    ## 498                                                    Shellard, Mr. Frederick William
    ## 499                                    Allison, Mrs. Hudson J C (Bessie Waldo Daniels)
    ## 500                                                                 Svensson, Mr. Olof
    ## 501                                                                   Calic, Mr. Petar
    ## 502                                                                Canavan, Miss. Mary
    ## 503                                                     O'Sullivan, Miss. Bridget Mary
    ## 504                                                     Laitinen, Miss. Kristina Sofia
    ## 505                                                              Maioni, Miss. Roberta
    ## 506                                         Penasco y Castellana, Mr. Victor de Satode
    ## 507                                      Quick, Mrs. Frederick Charles (Jane Richards)
    ## 508                                      Bradley, Mr. George ("George Arthur Brayton")
    ## 509                                                           Olsen, Mr. Henry Margido
    ## 510                                                                     Lang, Mr. Fang
    ## 511                                                           Daly, Mr. Eugene Patrick
    ## 512                                                                  Webber, Mr. James
    ## 513                                                          McGough, Mr. James Robert
    ## 514                                     Rothschild, Mrs. Martin (Elizabeth L. Barrett)
    ## 515                                                                  Coleff, Mr. Satio
    ## 516                                                       Walker, Mr. William Anderson
    ## 517                                                       Lemore, Mrs. (Amelia Milley)
    ## 518                                                                  Ryan, Mr. Patrick
    ## 519                               Angle, Mrs. William A (Florence "Mary" Agnes Hughes)
    ## 520                                                                Pavlovic, Mr. Stefo
    ## 521                                                              Perreault, Miss. Anne
    ## 522                                                                    Vovk, Mr. Janko
    ## 523                                                                 Lahoud, Mr. Sarkis
    ## 524                                    Hippach, Mrs. Louis Albert (Ida Sophia Fischer)
    ## 525                                                                  Kassem, Mr. Fared
    ## 526                                                                 Farrell, Mr. James
    ## 527                                                               Ridsdale, Miss. Lucy
    ## 528                                                                 Farthing, Mr. John
    ## 529                                                          Salonen, Mr. Johan Werner
    ## 530                                                        Hocking, Mr. Richard George
    ## 531                                                           Quick, Miss. Phyllis May
    ## 532                                                                  Toufik, Mr. Nakli
    ## 533                                                               Elias, Mr. Joseph Jr
    ## 534                                             Peter, Mrs. Catherine (Catherine Rizk)
    ## 535                                                                Cacic, Miss. Marija
    ## 536                                                             Hart, Miss. Eva Miriam
    ## 537                                                  Butt, Major. Archibald Willingham
    ## 538                                                                LeRoy, Miss. Bertha
    ## 539                                                           Risien, Mr. Samuel Beard
    ## 540                                                 Frolicher, Miss. Hedwig Margaritha
    ## 541                                                            Crosby, Miss. Harriet R
    ## 542                                               Andersson, Miss. Ingeborg Constanzia
    ## 543                                                  Andersson, Miss. Sigrid Elisabeth
    ## 544                                                                  Beane, Mr. Edward
    ## 545                                                         Douglas, Mr. Walter Donald
    ## 546                                                       Nicholson, Mr. Arthur Ernest
    ## 547                                                  Beane, Mrs. Edward (Ethel Clarke)
    ## 548                                                         Padro y Manent, Mr. Julian
    ## 549                                                          Goldsmith, Mr. Frank John
    ## 550                                                     Davies, Master. John Morgan Jr
    ## 551                                                        Thayer, Mr. John Borland Jr
    ## 552                                                        Sharp, Mr. Percival James R
    ## 553                                                               O'Brien, Mr. Timothy
    ## 554                                                  Leeni, Mr. Fahim ("Philip Zenni")
    ## 555                                                                 Ohman, Miss. Velin
    ## 556                                                                 Wright, Mr. George
    ## 557                  Duff Gordon, Lady. (Lucille Christiana Sutherland) ("Mrs Morgan")
    ## 558                                                                Robbins, Mr. Victor
    ## 559                                             Taussig, Mrs. Emil (Tillie Mandelbaum)
    ## 560                                       de Messemaeker, Mrs. Guillaume Joseph (Emma)
    ## 561                                                           Morrow, Mr. Thomas Rowan
    ## 562                                                                  Sivic, Mr. Husein
    ## 563                                                         Norman, Mr. Robert Douglas
    ## 564                                                                  Simmons, Mr. John
    ## 565                                                     Meanwell, Miss. (Marion Ogden)
    ## 566                                                               Davies, Mr. Alfred J
    ## 567                                                               Stoytcheff, Mr. Ilia
    ## 568                                        Palsson, Mrs. Nils (Alma Cornelia Berglund)
    ## 569                                                                Doharr, Mr. Tannous
    ## 570                                                                  Jonsson, Mr. Carl
    ## 571                                                                 Harris, Mr. George
    ## 572                                      Appleton, Mrs. Edward Dale (Charlotte Lamson)
    ## 573                                                   Flynn, Mr. John Irwin ("Irving")
    ## 574                                                                  Kelly, Miss. Mary
    ## 575                                                       Rush, Mr. Alfred George John
    ## 576                                                               Patchett, Mr. George
    ## 577                                                               Garside, Miss. Ethel
    ## 578                                          Silvey, Mrs. William Baird (Alice Munger)
    ## 579                                                   Caram, Mrs. Joseph (Maria Elias)
    ## 580                                                                Jussila, Mr. Eiriik
    ## 581                                                        Christy, Miss. Julie Rachel
    ## 582                               Thayer, Mrs. John Borland (Marian Longstreth Morris)
    ## 583                                                         Downton, Mr. William James
    ## 584                                                                Ross, Mr. John Hugo
    ## 585                                                                Paulner, Mr. Uscher
    ## 586                                                                Taussig, Miss. Ruth
    ## 587                                                            Jarvis, Mr. John Denzil
    ## 588                                                   Frolicher-Stehli, Mr. Maxmillian
    ## 589                                                              Gilinski, Mr. Eliezer
    ## 590                                                                Murdlin, Mr. Joseph
    ## 591                                                               Rintamaki, Mr. Matti
    ## 592                                    Stephenson, Mrs. Walter Bertram (Martha Eustis)
    ## 593                                                         Elsbury, Mr. William James
    ## 594                                                                 Bourke, Miss. Mary
    ## 595                                                            Chapman, Mr. John Henry
    ## 596                                                        Van Impe, Mr. Jean Baptiste
    ## 597                                                         Leitch, Miss. Jessie Wills
    ## 598                                                                Johnson, Mr. Alfred
    ## 599                                                                  Boulos, Mr. Hanna
    ## 600                                       Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")
    ## 601                                Jacobsohn, Mrs. Sidney Samuel (Amy Frances Christy)
    ## 602                                                               Slabenoff, Mr. Petco
    ## 603                                                          Harrington, Mr. Charles H
    ## 604                                                          Torber, Mr. Ernst William
    ## 605                                                    Homer, Mr. Harry ("Mr E Haven")
    ## 606                                                      Lindell, Mr. Edvard Bengtsson
    ## 607                                                                  Karaic, Mr. Milan
    ## 608                                                        Daniel, Mr. Robert Williams
    ## 609                              Laroche, Mrs. Joseph (Juliette Marie Louise Lafargue)
    ## 610                                                          Shutes, Miss. Elizabeth W
    ## 611                          Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren)
    ## 612                                                              Jardin, Mr. Jose Neto
    ## 613                                                        Murphy, Miss. Margaret Jane
    ## 614                                                                   Horgan, Mr. John
    ## 615                                                    Brocklebank, Mr. William Alfred
    ## 616                                                                Herman, Miss. Alice
    ## 617                                                          Danbom, Mr. Ernst Gilbert
    ## 618                                    Lobb, Mrs. William Arthur (Cordelia K Stanlick)
    ## 619                                                        Becker, Miss. Marion Louise
    ## 620                                                                Gavey, Mr. Lawrence
    ## 621                                                                Yasbeck, Mr. Antoni
    ## 622                                                       Kimball, Mr. Edwin Nelson Jr
    ## 623                                                                   Nakid, Mr. Sahid
    ## 624                                                        Hansen, Mr. Henry Damsgaard
    ## 625                                                        Bowen, Mr. David John "Dai"
    ## 626                                                              Sutton, Mr. Frederick
    ## 627                                                     Kirkland, Rev. Charles Leonard
    ## 628                                                      Longley, Miss. Gretchen Fiske
    ## 629                                                          Bostandyeff, Mr. Guentcho
    ## 630                                                           O'Connell, Mr. Patrick D
    ## 631                                               Barkworth, Mr. Algernon Henry Wilson
    ## 632                                                        Lundahl, Mr. Johan Svensson
    ## 633                                                          Stahelin-Maeglin, Dr. Max
    ## 634                                                      Parr, Mr. William Henry Marsh
    ## 635                                                                 Skoog, Miss. Mabel
    ## 636                                                                  Davis, Miss. Mary
    ## 637                                                         Leinonen, Mr. Antti Gustaf
    ## 638                                                                Collyer, Mr. Harvey
    ## 639                                             Panula, Mrs. Juha (Maria Emilia Ojala)
    ## 640                                                         Thorneycroft, Mr. Percival
    ## 641                                                             Jensen, Mr. Hans Peder
    ## 642                                                               Sagesser, Mlle. Emma
    ## 643                                                      Skoog, Miss. Margit Elizabeth
    ## 644                                                                    Foo, Mr. Choong
    ## 645                                                             Baclini, Miss. Eugenie
    ## 646                                                          Harper, Mr. Henry Sleeper
    ## 647                                                                  Cor, Mr. Liudevit
    ## 648                                                Simonius-Blumer, Col. Oberst Alfons
    ## 649                                                                 Willey, Mr. Edward
    ## 650                                                    Stanley, Miss. Amy Zillah Elsie
    ## 651                                                                  Mitkoff, Mr. Mito
    ## 652                                                                Doling, Miss. Elsie
    ## 653                                                     Kalvik, Mr. Johannes Halvorsen
    ## 654                                                      O'Leary, Miss. Hanora "Norah"
    ## 655                                                       Hegarty, Miss. Hanora "Nora"
    ## 656                                                          Hickman, Mr. Leonard Mark
    ## 657                                                              Radeff, Mr. Alexander
    ## 658                                                      Bourke, Mrs. John (Catherine)
    ## 659                                                       Eitemiller, Mr. George Floyd
    ## 660                                                         Newell, Mr. Arthur Webster
    ## 661                                                      Frauenthal, Dr. Henry William
    ## 662                                                                  Badt, Mr. Mohamed
    ## 663                                                         Colley, Mr. Edward Pomeroy
    ## 664                                                                   Coleff, Mr. Peju
    ## 665                                                        Lindqvist, Mr. Eino William
    ## 666                                                                 Hickman, Mr. Lewis
    ## 667                                                        Butler, Mr. Reginald Fenton
    ## 668                                                         Rommetvedt, Mr. Knud Paust
    ## 669                                                                    Cook, Mr. Jacob
    ## 670                                  Taylor, Mrs. Elmer Zebley (Juliet Cummins Wright)
    ## 671                      Brown, Mrs. Thomas William Solomon (Elizabeth Catherine Ford)
    ## 672                                                             Davidson, Mr. Thornton
    ## 673                                                        Mitchell, Mr. Henry Michael
    ## 674                                                              Wilhelms, Mr. Charles
    ## 675                                                         Watson, Mr. Ennis Hastings
    ## 676                                                     Edvardsson, Mr. Gustaf Hjalmar
    ## 677                                                      Sawyer, Mr. Frederick Charles
    ## 678                                                            Turja, Miss. Anna Sofia
    ## 679                                            Goodwin, Mrs. Frederick (Augusta Tyler)
    ## 680                                                 Cardeza, Mr. Thomas Drake Martinez
    ## 681                                                                Peters, Miss. Katie
    ## 682                                                                 Hassab, Mr. Hammad
    ## 683                                                        Olsvigen, Mr. Thor Anderson
    ## 684                                                        Goodwin, Mr. Charles Edward
    ## 685                                                  Brown, Mr. Thomas William Solomon
    ## 686                                             Laroche, Mr. Joseph Philippe Lemercier
    ## 687                                                           Panula, Mr. Jaako Arnold
    ## 688                                                                  Dakic, Mr. Branko
    ## 689                                                    Fischer, Mr. Eberhard Thelander
    ## 690                                                  Madill, Miss. Georgette Alexandra
    ## 691                                                            Dick, Mr. Albert Adrian
    ## 692                                                                 Karun, Miss. Manca
    ## 693                                                                       Lam, Mr. Ali
    ## 694                                                                   Saad, Mr. Khalil
    ## 695                                                                    Weir, Col. John
    ## 696                                                         Chapman, Mr. Charles Henry
    ## 697                                                                   Kelly, Mr. James
    ## 698                                                   Mullens, Miss. Katherine "Katie"
    ## 699                                                           Thayer, Mr. John Borland
    ## 700                                           Humblen, Mr. Adolf Mathias Nicolai Olsen
    ## 701                                  Astor, Mrs. John Jacob (Madeleine Talmadge Force)
    ## 702                                                   Silverthorne, Mr. Spencer Victor
    ## 703                                                              Barbara, Miss. Saiide
    ## 704                                                              Gallagher, Mr. Martin
    ## 705                                                            Hansen, Mr. Henrik Juul
    ## 706                                     Morley, Mr. Henry Samuel ("Mr Henry Marshall")
    ## 707                                                      Kelly, Mrs. Florence "Fannie"
    ## 708                                                  Calderhead, Mr. Edward Pennington
    ## 709                                                               Cleaver, Miss. Alice
    ## 710                                  Moubarek, Master. Halim Gonios ("William George")
    ## 711                                   Mayne, Mlle. Berthe Antonine ("Mrs de Villiers")
    ## 712                                                                 Klaber, Mr. Herman
    ## 713                                                           Taylor, Mr. Elmer Zebley
    ## 714                                                         Larsson, Mr. August Viktor
    ## 715                                                              Greenberg, Mr. Samuel
    ## 716                                         Soholt, Mr. Peter Andreas Lauritz Andersen
    ## 717                                                      Endres, Miss. Caroline Louise
    ## 718                                                Troutt, Miss. Edwina Celia "Winnie"
    ## 719                                                                McEvoy, Mr. Michael
    ## 720                                                       Johnson, Mr. Malkolm Joackim
    ## 721                                                  Harper, Miss. Annie Jessie "Nina"
    ## 722                                                          Jensen, Mr. Svend Lauritz
    ## 723                                                       Gillespie, Mr. William Henry
    ## 724                                                            Hodges, Mr. Henry Price
    ## 725                                                      Chambers, Mr. Norman Campbell
    ## 726                                                                Oreskovic, Mr. Luka
    ## 727                                        Renouf, Mrs. Peter Henry (Lillian Jefferys)
    ## 728                                                           Mannion, Miss. Margareth
    ## 729                                                    Bryhl, Mr. Kurt Arnold Gottfrid
    ## 730                                                      Ilmakangas, Miss. Pieta Sofia
    ## 731                                                      Allen, Miss. Elisabeth Walton
    ## 732                                                           Hassan, Mr. Houssein G N
    ## 733                                                               Knight, Mr. Robert J
    ## 734                                                         Berriman, Mr. William John
    ## 735                                                       Troupiansky, Mr. Moses Aaron
    ## 736                                                               Williams, Mr. Leslie
    ## 737                                            Ford, Mrs. Edward (Margaret Ann Watson)
    ## 738                                                             Lesurer, Mr. Gustave J
    ## 739                                                                 Ivanoff, Mr. Kanio
    ## 740                                                                 Nankoff, Mr. Minko
    ## 741                                                        Hawksford, Mr. Walter James
    ## 742                                                      Cavendish, Mr. Tyrell William
    ## 743                                              Ryerson, Miss. Susan Parker "Suzette"
    ## 744                                                                  McNamee, Mr. Neal
    ## 745                                                                 Stranden, Mr. Juho
    ## 746                                                       Crosby, Capt. Edward Gifford
    ## 747                                                        Abbott, Mr. Rossmore Edward
    ## 748                                                              Sinkkonen, Miss. Anna
    ## 749                                                          Marvin, Mr. Daniel Warner
    ## 750                                                            Connaghton, Mr. Michael
    ## 751                                                                  Wells, Miss. Joan
    ## 752                                                                Moor, Master. Meier
    ## 753                                                   Vande Velde, Mr. Johannes Joseph
    ## 754                                                                 Jonkoff, Mr. Lalio
    ## 755                                                   Herman, Mrs. Samuel (Jane Laver)
    ## 756                                                          Hamalainen, Master. Viljo
    ## 757                                                       Carlsson, Mr. August Sigfrid
    ## 758                                                           Bailey, Mr. Percy Andrew
    ## 759                                                       Theobald, Mr. Thomas Leonard
    ## 760                           Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)
    ## 761                                                                 Garfirth, Mr. John
    ## 762                                                     Nirva, Mr. Iisakki Antino Aijo
    ## 763                                                              Barah, Mr. Hanna Assi
    ## 764                                          Carter, Mrs. William Ernest (Lucile Polk)
    ## 765                                                             Eklund, Mr. Hans Linus
    ## 766                                               Hogeboom, Mrs. John C (Anna Andrews)
    ## 767                                                          Brewe, Dr. Arthur Jackson
    ## 768                                                                 Mangan, Miss. Mary
    ## 769                                                                Moran, Mr. Daniel J
    ## 770                                                   Gronnestad, Mr. Daniel Danielsen
    ## 771                                                             Lievens, Mr. Rene Aime
    ## 772                                                            Jensen, Mr. Niels Peder
    ## 773                                                                  Mack, Mrs. (Mary)
    ## 774                                                                    Elias, Mr. Dibo
    ## 775                                              Hocking, Mrs. Elizabeth (Eliza Needs)
    ## 776                                            Myhrman, Mr. Pehr Fabian Oliver Malkolm
    ## 777                                                                   Tobin, Mr. Roger
    ## 778                                                      Emanuel, Miss. Virginia Ethel
    ## 779                                                            Kilgannon, Mr. Thomas J
    ## 780                              Robert, Mrs. Edward Scott (Elisabeth Walton McMillan)
    ## 781                                                               Ayoub, Miss. Banoura
    ## 782                                          Dick, Mrs. Albert Adrian (Vera Gillespie)
    ## 783                                                             Long, Mr. Milton Clyde
    ## 784                                                             Johnston, Mr. Andrew G
    ## 785                                                                   Ali, Mr. William
    ## 786                                                 Harmer, Mr. Abraham (David Lishin)
    ## 787                                                          Sjoblom, Miss. Anna Sofia
    ## 788                                                          Rice, Master. George Hugh
    ## 789                                                         Dean, Master. Bertram Vere
    ## 790                                                           Guggenheim, Mr. Benjamin
    ## 791                                                           Keane, Mr. Andrew "Andy"
    ## 792                                                                Gaskell, Mr. Alfred
    ## 793                                                            Sage, Miss. Stella Anna
    ## 794                                                           Hoyt, Mr. William Fisher
    ## 795                                                              Dantcheff, Mr. Ristiu
    ## 796                                                                 Otter, Mr. Richard
    ## 797                                                        Leader, Dr. Alice (Farnham)
    ## 798                                                                   Osman, Mrs. Mara
    ## 799                                                       Ibrahim Shawah, Mr. Yousseff
    ## 800                               Van Impe, Mrs. Jean Baptiste (Rosalie Paula Govaert)
    ## 801                                                               Ponesell, Mr. Martin
    ## 802                                        Collyer, Mrs. Harvey (Charlotte Annie Tate)
    ## 803                                                Carter, Master. William Thornton II
    ## 804                                                    Thomas, Master. Assad Alexander
    ## 805                                                            Hedman, Mr. Oskar Arvid
    ## 806                                                          Johansson, Mr. Karl Johan
    ## 807                                                             Andrews, Mr. Thomas Jr
    ## 808                                                    Pettersson, Miss. Ellen Natalia
    ## 809                                                                  Meyer, Mr. August
    ## 810                                     Chambers, Mrs. Norman Campbell (Bertha Griggs)
    ## 811                                                             Alexander, Mr. William
    ## 812                                                                  Lester, Mr. James
    ## 813                                                          Slemen, Mr. Richard James
    ## 814                                                 Andersson, Miss. Ebba Iris Alfrida
    ## 815                                                         Tomlin, Mr. Ernest Portage
    ## 816                                                                   Fry, Mr. Richard
    ## 817                                                       Heininen, Miss. Wendla Maria
    ## 818                                                                 Mallet, Mr. Albert
    ## 819                                                   Holm, Mr. John Fredrik Alexander
    ## 820                                                       Skoog, Master. Karl Thorsten
    ## 821                                 Hays, Mrs. Charles Melville (Clara Jennings Gregg)
    ## 822                                                                  Lulic, Mr. Nikola
    ## 823                                                    Reuchlin, Jonkheer. John George
    ## 824                                                                 Moor, Mrs. (Beila)
    ## 825                                                       Panula, Master. Urho Abraham
    ## 826                                                                    Flynn, Mr. John
    ## 827                                                                       Lam, Mr. Len
    ## 828                                                              Mallet, Master. Andre
    ## 829                                                       McCormack, Mr. Thomas Joseph
    ## 830                                          Stone, Mrs. George Nelson (Martha Evelyn)
    ## 831                                            Yasbeck, Mrs. Antoni (Selini Alexander)
    ## 832                                                    Richards, Master. George Sibley
    ## 833                                                                     Saad, Mr. Amin
    ## 834                                                             Augustsson, Mr. Albert
    ## 835                                                             Allum, Mr. Owen George
    ## 836                                                        Compton, Miss. Sara Rebecca
    ## 837                                                                   Pasic, Mr. Jakob
    ## 838                                                                Sirota, Mr. Maurice
    ## 839                                                                    Chip, Mr. Chang
    ## 840                                                               Marechal, Mr. Pierre
    ## 841                                                        Alhomaki, Mr. Ilmari Rudolf
    ## 842                                                           Mudd, Mr. Thomas Charles
    ## 843                                                            Serepeca, Miss. Augusta
    ## 844                                                         Lemberopolous, Mr. Peter L
    ## 845                                                                Culumovic, Mr. Jeso
    ## 846                                                                Abbing, Mr. Anthony
    ## 847                                                           Sage, Mr. Douglas Bullen
    ## 848                                                                 Markoff, Mr. Marin
    ## 849                                                                  Harper, Rev. John
    ## 850                                       Goldenberg, Mrs. Samuel L (Edwiga Grabowska)
    ## 851                                            Andersson, Master. Sigvard Harald Elias
    ## 852                                                                Svensson, Mr. Johan
    ## 853                                                            Boulos, Miss. Nourelain
    ## 854                                                          Lines, Miss. Mary Conover
    ## 855                                      Carter, Mrs. Ernest Courtenay (Lilian Hughes)
    ## 856                                                         Aks, Mrs. Sam (Leah Rosen)
    ## 857                                         Wick, Mrs. George Dennick (Mary Hitchcock)
    ## 858                                                             Daly, Mr. Peter Denis 
    ## 859                                              Baclini, Mrs. Solomon (Latifa Qurban)
    ## 860                                                                   Razi, Mr. Raihed
    ## 861                                                            Hansen, Mr. Claus Peter
    ## 862                                                        Giles, Mr. Frederick Edward
    ## 863                                Swift, Mrs. Frederick Joel (Margaret Welles Barron)
    ## 864                                                  Sage, Miss. Dorothy Edith "Dolly"
    ## 865                                                             Gill, Mr. John William
    ## 866                                                           Bystrom, Mrs. (Karolina)
    ## 867                                                       Duran y More, Miss. Asuncion
    ## 868                                               Roebling, Mr. Washington Augustus II
    ## 869                                                        van Melkebeke, Mr. Philemon
    ## 870                                                    Johnson, Master. Harold Theodor
    ## 871                                                                  Balkic, Mr. Cerin
    ## 872                                   Beckwith, Mrs. Richard Leonard (Sallie Monypeny)
    ## 873                                                           Carlsson, Mr. Frans Olof
    ## 874                                                        Vander Cruyssen, Mr. Victor
    ## 875                                              Abelson, Mrs. Samuel (Hannah Wizosky)
    ## 876                                                   Najib, Miss. Adele Kiamie "Jane"
    ## 877                                                      Gustafsson, Mr. Alfred Ossian
    ## 878                                                               Petroff, Mr. Nedelio
    ## 879                                                                 Laleff, Mr. Kristo
    ## 880                                      Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)
    ## 881                                       Shelley, Mrs. William (Imanita Parrish Hall)
    ## 882                                                                 Markun, Mr. Johann
    ## 883                                                       Dahlberg, Miss. Gerda Ulrika
    ## 884                                                      Banfield, Mr. Frederick James
    ## 885                                                             Sutehall, Mr. Henry Jr
    ## 886                                               Rice, Mrs. William (Margaret Norton)
    ## 887                                                              Montvila, Rev. Juozas
    ## 888                                                       Graham, Miss. Margaret Edith
    ## 889                                           Johnston, Miss. Catherine Helen "Carrie"
    ## 890                                                              Behr, Mr. Karl Howell
    ## 891                                                                Dooley, Mr. Patrick
    ##        Sex   Age SibSp Parch             Ticket     Fare           Cabin
    ## 1     male 22.00     1     0          A/5 21171   7.2500                
    ## 2   female 38.00     1     0           PC 17599  71.2833             C85
    ## 3   female 26.00     0     0   STON/O2. 3101282   7.9250                
    ## 4   female 35.00     1     0             113803  53.1000            C123
    ## 5     male 35.00     0     0             373450   8.0500                
    ## 6     male    NA     0     0             330877   8.4583                
    ## 7     male 54.00     0     0              17463  51.8625             E46
    ## 8     male  2.00     3     1             349909  21.0750                
    ## 9   female 27.00     0     2             347742  11.1333                
    ## 10  female 14.00     1     0             237736  30.0708                
    ## 11  female  4.00     1     1            PP 9549  16.7000              G6
    ## 12  female 58.00     0     0             113783  26.5500            C103
    ## 13    male 20.00     0     0          A/5. 2151   8.0500                
    ## 14    male 39.00     1     5             347082  31.2750                
    ## 15  female 14.00     0     0             350406   7.8542                
    ## 16  female 55.00     0     0             248706  16.0000                
    ## 17    male  2.00     4     1             382652  29.1250                
    ## 18    male    NA     0     0             244373  13.0000                
    ## 19  female 31.00     1     0             345763  18.0000                
    ## 20  female    NA     0     0               2649   7.2250                
    ## 21    male 35.00     0     0             239865  26.0000                
    ## 22    male 34.00     0     0             248698  13.0000             D56
    ## 23  female 15.00     0     0             330923   8.0292                
    ## 24    male 28.00     0     0             113788  35.5000              A6
    ## 25  female  8.00     3     1             349909  21.0750                
    ## 26  female 38.00     1     5             347077  31.3875                
    ## 27    male    NA     0     0               2631   7.2250                
    ## 28    male 19.00     3     2              19950 263.0000     C23 C25 C27
    ## 29  female    NA     0     0             330959   7.8792                
    ## 30    male    NA     0     0             349216   7.8958                
    ## 31    male 40.00     0     0           PC 17601  27.7208                
    ## 32  female    NA     1     0           PC 17569 146.5208             B78
    ## 33  female    NA     0     0             335677   7.7500                
    ## 34    male 66.00     0     0         C.A. 24579  10.5000                
    ## 35    male 28.00     1     0           PC 17604  82.1708                
    ## 36    male 42.00     1     0             113789  52.0000                
    ## 37    male    NA     0     0               2677   7.2292                
    ## 38    male 21.00     0     0         A./5. 2152   8.0500                
    ## 39  female 18.00     2     0             345764  18.0000                
    ## 40  female 14.00     1     0               2651  11.2417                
    ## 41  female 40.00     1     0               7546   9.4750                
    ## 42  female 27.00     1     0              11668  21.0000                
    ## 43    male    NA     0     0             349253   7.8958                
    ## 44  female  3.00     1     2      SC/Paris 2123  41.5792                
    ## 45  female 19.00     0     0             330958   7.8792                
    ## 46    male    NA     0     0    S.C./A.4. 23567   8.0500                
    ## 47    male    NA     1     0             370371  15.5000                
    ## 48  female    NA     0     0              14311   7.7500                
    ## 49    male    NA     2     0               2662  21.6792                
    ## 50  female 18.00     1     0             349237  17.8000                
    ## 51    male  7.00     4     1            3101295  39.6875                
    ## 52    male 21.00     0     0         A/4. 39886   7.8000                
    ## 53  female 49.00     1     0           PC 17572  76.7292             D33
    ## 54  female 29.00     1     0               2926  26.0000                
    ## 55    male 65.00     0     1             113509  61.9792             B30
    ## 56    male    NA     0     0              19947  35.5000             C52
    ## 57  female 21.00     0     0         C.A. 31026  10.5000                
    ## 58    male 28.50     0     0               2697   7.2292                
    ## 59  female  5.00     1     2         C.A. 34651  27.7500                
    ## 60    male 11.00     5     2            CA 2144  46.9000                
    ## 61    male 22.00     0     0               2669   7.2292                
    ## 62  female 38.00     0     0             113572  80.0000             B28
    ## 63    male 45.00     1     0              36973  83.4750             C83
    ## 64    male  4.00     3     2             347088  27.9000                
    ## 65    male    NA     0     0           PC 17605  27.7208                
    ## 66    male    NA     1     1               2661  15.2458                
    ## 67  female 29.00     0     0         C.A. 29395  10.5000             F33
    ## 68    male 19.00     0     0          S.P. 3464   8.1583                
    ## 69  female 17.00     4     2            3101281   7.9250                
    ## 70    male 26.00     2     0             315151   8.6625                
    ## 71    male 32.00     0     0         C.A. 33111  10.5000                
    ## 72  female 16.00     5     2            CA 2144  46.9000                
    ## 73    male 21.00     0     0       S.O.C. 14879  73.5000                
    ## 74    male 26.00     1     0               2680  14.4542                
    ## 75    male 32.00     0     0               1601  56.4958                
    ## 76    male 25.00     0     0             348123   7.6500           F G73
    ## 77    male    NA     0     0             349208   7.8958                
    ## 78    male    NA     0     0             374746   8.0500                
    ## 79    male  0.83     0     2             248738  29.0000                
    ## 80  female 30.00     0     0             364516  12.4750                
    ## 81    male 22.00     0     0             345767   9.0000                
    ## 82    male 29.00     0     0             345779   9.5000                
    ## 83  female    NA     0     0             330932   7.7875                
    ## 84    male 28.00     0     0             113059  47.1000                
    ## 85  female 17.00     0     0         SO/C 14885  10.5000                
    ## 86  female 33.00     3     0            3101278  15.8500                
    ## 87    male 16.00     1     3         W./C. 6608  34.3750                
    ## 88    male    NA     0     0    SOTON/OQ 392086   8.0500                
    ## 89  female 23.00     3     2              19950 263.0000     C23 C25 C27
    ## 90    male 24.00     0     0             343275   8.0500                
    ## 91    male 29.00     0     0             343276   8.0500                
    ## 92    male 20.00     0     0             347466   7.8542                
    ## 93    male 46.00     1     0        W.E.P. 5734  61.1750             E31
    ## 94    male 26.00     1     2          C.A. 2315  20.5750                
    ## 95    male 59.00     0     0             364500   7.2500                
    ## 96    male    NA     0     0             374910   8.0500                
    ## 97    male 71.00     0     0           PC 17754  34.6542              A5
    ## 98    male 23.00     0     1           PC 17759  63.3583         D10 D12
    ## 99  female 34.00     0     1             231919  23.0000                
    ## 100   male 34.00     1     0             244367  26.0000                
    ## 101 female 28.00     0     0             349245   7.8958                
    ## 102   male    NA     0     0             349215   7.8958                
    ## 103   male 21.00     0     1              35281  77.2875             D26
    ## 104   male 33.00     0     0               7540   8.6542                
    ## 105   male 37.00     2     0            3101276   7.9250                
    ## 106   male 28.00     0     0             349207   7.8958                
    ## 107 female 21.00     0     0             343120   7.6500                
    ## 108   male    NA     0     0             312991   7.7750                
    ## 109   male 38.00     0     0             349249   7.8958                
    ## 110 female    NA     1     0             371110  24.1500                
    ## 111   male 47.00     0     0             110465  52.0000            C110
    ## 112 female 14.50     1     0               2665  14.4542                
    ## 113   male 22.00     0     0             324669   8.0500                
    ## 114 female 20.00     1     0               4136   9.8250                
    ## 115 female 17.00     0     0               2627  14.4583                
    ## 116   male 21.00     0     0  STON/O 2. 3101294   7.9250                
    ## 117   male 70.50     0     0             370369   7.7500                
    ## 118   male 29.00     1     0              11668  21.0000                
    ## 119   male 24.00     0     1           PC 17558 247.5208         B58 B60
    ## 120 female  2.00     4     2             347082  31.2750                
    ## 121   male 21.00     2     0       S.O.C. 14879  73.5000                
    ## 122   male    NA     0     0          A4. 54510   8.0500                
    ## 123   male 32.50     1     0             237736  30.0708                
    ## 124 female 32.50     0     0              27267  13.0000            E101
    ## 125   male 54.00     0     1              35281  77.2875             D26
    ## 126   male 12.00     1     0               2651  11.2417                
    ## 127   male    NA     0     0             370372   7.7500                
    ## 128   male 24.00     0     0            C 17369   7.1417                
    ## 129 female    NA     1     1               2668  22.3583           F E69
    ## 130   male 45.00     0     0             347061   6.9750                
    ## 131   male 33.00     0     0             349241   7.8958                
    ## 132   male 20.00     0     0 SOTON/O.Q. 3101307   7.0500                
    ## 133 female 47.00     1     0          A/5. 3337  14.5000                
    ## 134 female 29.00     1     0             228414  26.0000                
    ## 135   male 25.00     0     0         C.A. 29178  13.0000                
    ## 136   male 23.00     0     0      SC/PARIS 2133  15.0458                
    ## 137 female 19.00     0     2              11752  26.2833             D47
    ## 138   male 37.00     1     0             113803  53.1000            C123
    ## 139   male 16.00     0     0               7534   9.2167                
    ## 140   male 24.00     0     0           PC 17593  79.2000             B86
    ## 141 female    NA     0     2               2678  15.2458                
    ## 142 female 22.00     0     0             347081   7.7500                
    ## 143 female 24.00     1     0   STON/O2. 3101279  15.8500                
    ## 144   male 19.00     0     0             365222   6.7500                
    ## 145   male 18.00     0     0             231945  11.5000                
    ## 146   male 19.00     1     1         C.A. 33112  36.7500                
    ## 147   male 27.00     0     0             350043   7.7958                
    ## 148 female  9.00     2     2         W./C. 6608  34.3750                
    ## 149   male 36.50     0     2             230080  26.0000              F2
    ## 150   male 42.00     0     0             244310  13.0000                
    ## 151   male 51.00     0     0        S.O.P. 1166  12.5250                
    ## 152 female 22.00     1     0             113776  66.6000              C2
    ## 153   male 55.50     0     0         A.5. 11206   8.0500                
    ## 154   male 40.50     0     2           A/5. 851  14.5000                
    ## 155   male    NA     0     0          Fa 265302   7.3125                
    ## 156   male 51.00     0     1           PC 17597  61.3792                
    ## 157 female 16.00     0     0              35851   7.7333                
    ## 158   male 30.00     0     0    SOTON/OQ 392090   8.0500                
    ## 159   male    NA     0     0             315037   8.6625                
    ## 160   male    NA     8     2           CA. 2343  69.5500                
    ## 161   male 44.00     0     1             371362  16.1000                
    ## 162 female 40.00     0     0         C.A. 33595  15.7500                
    ## 163   male 26.00     0     0             347068   7.7750                
    ## 164   male 17.00     0     0             315093   8.6625                
    ## 165   male  1.00     4     1            3101295  39.6875                
    ## 166   male  9.00     0     2             363291  20.5250                
    ## 167 female    NA     0     1             113505  55.0000             E33
    ## 168 female 45.00     1     4             347088  27.9000                
    ## 169   male    NA     0     0           PC 17318  25.9250                
    ## 170   male 28.00     0     0               1601  56.4958                
    ## 171   male 61.00     0     0             111240  33.5000             B19
    ## 172   male  4.00     4     1             382652  29.1250                
    ## 173 female  1.00     1     1             347742  11.1333                
    ## 174   male 21.00     0     0  STON/O 2. 3101280   7.9250                
    ## 175   male 56.00     0     0              17764  30.6958              A7
    ## 176   male 18.00     1     1             350404   7.8542                
    ## 177   male    NA     3     1               4133  25.4667                
    ## 178 female 50.00     0     0           PC 17595  28.7125             C49
    ## 179   male 30.00     0     0             250653  13.0000                
    ## 180   male 36.00     0     0               LINE   0.0000                
    ## 181 female    NA     8     2           CA. 2343  69.5500                
    ## 182   male    NA     0     0      SC/PARIS 2131  15.0500                
    ## 183   male  9.00     4     2             347077  31.3875                
    ## 184   male  1.00     2     1             230136  39.0000              F4
    ## 185 female  4.00     0     2             315153  22.0250                
    ## 186   male    NA     0     0             113767  50.0000             A32
    ## 187 female    NA     1     0             370365  15.5000                
    ## 188   male 45.00     0     0             111428  26.5500                
    ## 189   male 40.00     1     1             364849  15.5000                
    ## 190   male 36.00     0     0             349247   7.8958                
    ## 191 female 32.00     0     0             234604  13.0000                
    ## 192   male 19.00     0     0              28424  13.0000                
    ## 193 female 19.00     1     0             350046   7.8542                
    ## 194   male  3.00     1     1             230080  26.0000              F2
    ## 195 female 44.00     0     0           PC 17610  27.7208              B4
    ## 196 female 58.00     0     0           PC 17569 146.5208             B80
    ## 197   male    NA     0     0             368703   7.7500                
    ## 198   male 42.00     0     1               4579   8.4042                
    ## 199 female    NA     0     0             370370   7.7500                
    ## 200 female 24.00     0     0             248747  13.0000                
    ## 201   male 28.00     0     0             345770   9.5000                
    ## 202   male    NA     8     2           CA. 2343  69.5500                
    ## 203   male 34.00     0     0            3101264   6.4958                
    ## 204   male 45.50     0     0               2628   7.2250                
    ## 205   male 18.00     0     0           A/5 3540   8.0500                
    ## 206 female  2.00     0     1             347054  10.4625              G6
    ## 207   male 32.00     1     0            3101278  15.8500                
    ## 208   male 26.00     0     0               2699  18.7875                
    ## 209 female 16.00     0     0             367231   7.7500                
    ## 210   male 40.00     0     0             112277  31.0000             A31
    ## 211   male 24.00     0     0 SOTON/O.Q. 3101311   7.0500                
    ## 212 female 35.00     0     0       F.C.C. 13528  21.0000                
    ## 213   male 22.00     0     0          A/5 21174   7.2500                
    ## 214   male 30.00     0     0             250646  13.0000                
    ## 215   male    NA     1     0             367229   7.7500                
    ## 216 female 31.00     1     0              35273 113.2750             D36
    ## 217 female 27.00     0     0   STON/O2. 3101283   7.9250                
    ## 218   male 42.00     1     0             243847  27.0000                
    ## 219 female 32.00     0     0              11813  76.2917             D15
    ## 220   male 30.00     0     0          W/C 14208  10.5000                
    ## 221   male 16.00     0     0    SOTON/OQ 392089   8.0500                
    ## 222   male 27.00     0     0             220367  13.0000                
    ## 223   male 51.00     0     0              21440   8.0500                
    ## 224   male    NA     0     0             349234   7.8958                
    ## 225   male 38.00     1     0              19943  90.0000             C93
    ## 226   male 22.00     0     0            PP 4348   9.3500                
    ## 227   male 19.00     0     0          SW/PP 751  10.5000                
    ## 228   male 20.50     0     0          A/5 21173   7.2500                
    ## 229   male 18.00     0     0             236171  13.0000                
    ## 230 female    NA     3     1               4133  25.4667                
    ## 231 female 35.00     1     0              36973  83.4750             C83
    ## 232   male 29.00     0     0             347067   7.7750                
    ## 233   male 59.00     0     0             237442  13.5000                
    ## 234 female  5.00     4     2             347077  31.3875                
    ## 235   male 24.00     0     0         C.A. 29566  10.5000                
    ## 236 female    NA     0     0         W./C. 6609   7.5500                
    ## 237   male 44.00     1     0              26707  26.0000                
    ## 238 female  8.00     0     2         C.A. 31921  26.2500                
    ## 239   male 19.00     0     0              28665  10.5000                
    ## 240   male 33.00     0     0         SCO/W 1585  12.2750                
    ## 241 female    NA     1     0               2665  14.4542                
    ## 242 female    NA     1     0             367230  15.5000                
    ## 243   male 29.00     0     0        W./C. 14263  10.5000                
    ## 244   male 22.00     0     0  STON/O 2. 3101275   7.1250                
    ## 245   male 30.00     0     0               2694   7.2250                
    ## 246   male 44.00     2     0              19928  90.0000             C78
    ## 247 female 25.00     0     0             347071   7.7750                
    ## 248 female 24.00     0     2             250649  14.5000                
    ## 249   male 37.00     1     1              11751  52.5542             D35
    ## 250   male 54.00     1     0             244252  26.0000                
    ## 251   male    NA     0     0             362316   7.2500                
    ## 252 female 29.00     1     1             347054  10.4625              G6
    ## 253   male 62.00     0     0             113514  26.5500             C87
    ## 254   male 30.00     1     0          A/5. 3336  16.1000                
    ## 255 female 41.00     0     2             370129  20.2125                
    ## 256 female 29.00     0     2               2650  15.2458                
    ## 257 female    NA     0     0           PC 17585  79.2000                
    ## 258 female 30.00     0     0             110152  86.5000             B77
    ## 259 female 35.00     0     0           PC 17755 512.3292                
    ## 260 female 50.00     0     1             230433  26.0000                
    ## 261   male    NA     0     0             384461   7.7500                
    ## 262   male  3.00     4     2             347077  31.3875                
    ## 263   male 52.00     1     1             110413  79.6500             E67
    ## 264   male 40.00     0     0             112059   0.0000             B94
    ## 265 female    NA     0     0             382649   7.7500                
    ## 266   male 36.00     0     0         C.A. 17248  10.5000                
    ## 267   male 16.00     4     1            3101295  39.6875                
    ## 268   male 25.00     1     0             347083   7.7750                
    ## 269 female 58.00     0     1           PC 17582 153.4625            C125
    ## 270 female 35.00     0     0           PC 17760 135.6333             C99
    ## 271   male    NA     0     0             113798  31.0000                
    ## 272   male 25.00     0     0               LINE   0.0000                
    ## 273 female 41.00     0     1             250644  19.5000                
    ## 274   male 37.00     0     1           PC 17596  29.7000            C118
    ## 275 female    NA     0     0             370375   7.7500                
    ## 276 female 63.00     1     0              13502  77.9583              D7
    ## 277 female 45.00     0     0             347073   7.7500                
    ## 278   male    NA     0     0             239853   0.0000                
    ## 279   male  7.00     4     1             382652  29.1250                
    ## 280 female 35.00     1     1          C.A. 2673  20.2500                
    ## 281   male 65.00     0     0             336439   7.7500                
    ## 282   male 28.00     0     0             347464   7.8542                
    ## 283   male 16.00     0     0             345778   9.5000                
    ## 284   male 19.00     0     0         A/5. 10482   8.0500                
    ## 285   male    NA     0     0             113056  26.0000             A19
    ## 286   male 33.00     0     0             349239   8.6625                
    ## 287   male 30.00     0     0             345774   9.5000                
    ## 288   male 22.00     0     0             349206   7.8958                
    ## 289   male 42.00     0     0             237798  13.0000                
    ## 290 female 22.00     0     0             370373   7.7500                
    ## 291 female 26.00     0     0              19877  78.8500                
    ## 292 female 19.00     1     0              11967  91.0792             B49
    ## 293   male 36.00     0     0      SC/Paris 2163  12.8750               D
    ## 294 female 24.00     0     0             349236   8.8500                
    ## 295   male 24.00     0     0             349233   7.8958                
    ## 296   male    NA     0     0           PC 17612  27.7208                
    ## 297   male 23.50     0     0               2693   7.2292                
    ## 298 female  2.00     1     2             113781 151.5500         C22 C26
    ## 299   male    NA     0     0              19988  30.5000            C106
    ## 300 female 50.00     0     1           PC 17558 247.5208         B58 B60
    ## 301 female    NA     0     0               9234   7.7500                
    ## 302   male    NA     2     0             367226  23.2500                
    ## 303   male 19.00     0     0               LINE   0.0000                
    ## 304 female    NA     0     0             226593  12.3500            E101
    ## 305   male    NA     0     0           A/5 2466   8.0500                
    ## 306   male  0.92     1     2             113781 151.5500         C22 C26
    ## 307 female    NA     0     0              17421 110.8833                
    ## 308 female 17.00     1     0           PC 17758 108.9000             C65
    ## 309   male 30.00     1     0          P/PP 3381  24.0000                
    ## 310 female 30.00     0     0           PC 17485  56.9292             E36
    ## 311 female 24.00     0     0              11767  83.1583             C54
    ## 312 female 18.00     2     2           PC 17608 262.3750 B57 B59 B63 B66
    ## 313 female 26.00     1     1             250651  26.0000                
    ## 314   male 28.00     0     0             349243   7.8958                
    ## 315   male 43.00     1     1       F.C.C. 13529  26.2500                
    ## 316 female 26.00     0     0             347470   7.8542                
    ## 317 female 24.00     1     0             244367  26.0000                
    ## 318   male 54.00     0     0              29011  14.0000                
    ## 319 female 31.00     0     2              36928 164.8667              C7
    ## 320 female 40.00     1     1              16966 134.5000             E34
    ## 321   male 22.00     0     0          A/5 21172   7.2500                
    ## 322   male 27.00     0     0             349219   7.8958                
    ## 323 female 30.00     0     0             234818  12.3500                
    ## 324 female 22.00     1     1             248738  29.0000                
    ## 325   male    NA     8     2           CA. 2343  69.5500                
    ## 326 female 36.00     0     0           PC 17760 135.6333             C32
    ## 327   male 61.00     0     0             345364   6.2375                
    ## 328 female 36.00     0     0              28551  13.0000               D
    ## 329 female 31.00     1     1             363291  20.5250                
    ## 330 female 16.00     0     1             111361  57.9792             B18
    ## 331 female    NA     2     0             367226  23.2500                
    ## 332   male 45.50     0     0             113043  28.5000            C124
    ## 333   male 38.00     0     1           PC 17582 153.4625             C91
    ## 334   male 16.00     2     0             345764  18.0000                
    ## 335 female    NA     1     0           PC 17611 133.6500                
    ## 336   male    NA     0     0             349225   7.8958                
    ## 337   male 29.00     1     0             113776  66.6000              C2
    ## 338 female 41.00     0     0              16966 134.5000             E40
    ## 339   male 45.00     0     0               7598   8.0500                
    ## 340   male 45.00     0     0             113784  35.5000               T
    ## 341   male  2.00     1     1             230080  26.0000              F2
    ## 342 female 24.00     3     2              19950 263.0000     C23 C25 C27
    ## 343   male 28.00     0     0             248740  13.0000                
    ## 344   male 25.00     0     0             244361  13.0000                
    ## 345   male 36.00     0     0             229236  13.0000                
    ## 346 female 24.00     0     0             248733  13.0000             F33
    ## 347 female 40.00     0     0              31418  13.0000                
    ## 348 female    NA     1     0             386525  16.1000                
    ## 349   male  3.00     1     1         C.A. 37671  15.9000                
    ## 350   male 42.00     0     0             315088   8.6625                
    ## 351   male 23.00     0     0               7267   9.2250                
    ## 352   male    NA     0     0             113510  35.0000            C128
    ## 353   male 15.00     1     1               2695   7.2292                
    ## 354   male 25.00     1     0             349237  17.8000                
    ## 355   male    NA     0     0               2647   7.2250                
    ## 356   male 28.00     0     0             345783   9.5000                
    ## 357 female 22.00     0     1             113505  55.0000             E33
    ## 358 female 38.00     0     0             237671  13.0000                
    ## 359 female    NA     0     0             330931   7.8792                
    ## 360 female    NA     0     0             330980   7.8792                
    ## 361   male 40.00     1     4             347088  27.9000                
    ## 362   male 29.00     1     0      SC/PARIS 2167  27.7208                
    ## 363 female 45.00     0     1               2691  14.4542                
    ## 364   male 35.00     0     0 SOTON/O.Q. 3101310   7.0500                
    ## 365   male    NA     1     0             370365  15.5000                
    ## 366   male 30.00     0     0             C 7076   7.2500                
    ## 367 female 60.00     1     0             110813  75.2500             D37
    ## 368 female    NA     0     0               2626   7.2292                
    ## 369 female    NA     0     0              14313   7.7500                
    ## 370 female 24.00     0     0           PC 17477  69.3000             B35
    ## 371   male 25.00     1     0              11765  55.4417             E50
    ## 372   male 18.00     1     0            3101267   6.4958                
    ## 373   male 19.00     0     0             323951   8.0500                
    ## 374   male 22.00     0     0           PC 17760 135.6333                
    ## 375 female  3.00     3     1             349909  21.0750                
    ## 376 female    NA     1     0           PC 17604  82.1708                
    ## 377 female 22.00     0     0             C 7077   7.2500                
    ## 378   male 27.00     0     2             113503 211.5000             C82
    ## 379   male 20.00     0     0               2648   4.0125                
    ## 380   male 19.00     0     0             347069   7.7750                
    ## 381 female 42.00     0     0           PC 17757 227.5250                
    ## 382 female  1.00     0     2               2653  15.7417                
    ## 383   male 32.00     0     0  STON/O 2. 3101293   7.9250                
    ## 384 female 35.00     1     0             113789  52.0000                
    ## 385   male    NA     0     0             349227   7.8958                
    ## 386   male 18.00     0     0       S.O.C. 14879  73.5000                
    ## 387   male  1.00     5     2            CA 2144  46.9000                
    ## 388 female 36.00     0     0              27849  13.0000                
    ## 389   male    NA     0     0             367655   7.7292                
    ## 390 female 17.00     0     0            SC 1748  12.0000                
    ## 391   male 36.00     1     2             113760 120.0000         B96 B98
    ## 392   male 21.00     0     0             350034   7.7958                
    ## 393   male 28.00     2     0            3101277   7.9250                
    ## 394 female 23.00     1     0              35273 113.2750             D36
    ## 395 female 24.00     0     2            PP 9549  16.7000              G6
    ## 396   male 22.00     0     0             350052   7.7958                
    ## 397 female 31.00     0     0             350407   7.8542                
    ## 398   male 46.00     0     0              28403  26.0000                
    ## 399   male 23.00     0     0             244278  10.5000                
    ## 400 female 28.00     0     0             240929  12.6500                
    ## 401   male 39.00     0     0  STON/O 2. 3101289   7.9250                
    ## 402   male 26.00     0     0             341826   8.0500                
    ## 403 female 21.00     1     0               4137   9.8250                
    ## 404   male 28.00     1     0   STON/O2. 3101279  15.8500                
    ## 405 female 20.00     0     0             315096   8.6625                
    ## 406   male 34.00     1     0              28664  21.0000                
    ## 407   male 51.00     0     0             347064   7.7500                
    ## 408   male  3.00     1     1              29106  18.7500                
    ## 409   male 21.00     0     0             312992   7.7750                
    ## 410 female    NA     3     1               4133  25.4667                
    ## 411   male    NA     0     0             349222   7.8958                
    ## 412   male    NA     0     0             394140   6.8583                
    ## 413 female 33.00     1     0              19928  90.0000             C78
    ## 414   male    NA     0     0             239853   0.0000                
    ## 415   male 44.00     0     0  STON/O 2. 3101269   7.9250                
    ## 416 female    NA     0     0             343095   8.0500                
    ## 417 female 34.00     1     1              28220  32.5000                
    ## 418 female 18.00     0     2             250652  13.0000                
    ## 419   male 30.00     0     0              28228  13.0000                
    ## 420 female 10.00     0     2             345773  24.1500                
    ## 421   male    NA     0     0             349254   7.8958                
    ## 422   male 21.00     0     0         A/5. 13032   7.7333                
    ## 423   male 29.00     0     0             315082   7.8750                
    ## 424 female 28.00     1     1             347080  14.4000                
    ## 425   male 18.00     1     1             370129  20.2125                
    ## 426   male    NA     0     0         A/4. 34244   7.2500                
    ## 427 female 28.00     1     0               2003  26.0000                
    ## 428 female 19.00     0     0             250655  26.0000                
    ## 429   male    NA     0     0             364851   7.7500                
    ## 430   male 32.00     0     0  SOTON/O.Q. 392078   8.0500             E10
    ## 431   male 28.00     0     0             110564  26.5500             C52
    ## 432 female    NA     1     0             376564  16.1000                
    ## 433 female 42.00     1     0         SC/AH 3085  26.0000                
    ## 434   male 17.00     0     0  STON/O 2. 3101274   7.1250                
    ## 435   male 50.00     1     0              13507  55.9000             E44
    ## 436 female 14.00     1     2             113760 120.0000         B96 B98
    ## 437 female 21.00     2     2         W./C. 6608  34.3750                
    ## 438 female 24.00     2     3              29106  18.7500                
    ## 439   male 64.00     1     4              19950 263.0000     C23 C25 C27
    ## 440   male 31.00     0     0         C.A. 18723  10.5000                
    ## 441 female 45.00     1     1       F.C.C. 13529  26.2500                
    ## 442   male 20.00     0     0             345769   9.5000                
    ## 443   male 25.00     1     0             347076   7.7750                
    ## 444 female 28.00     0     0             230434  13.0000                
    ## 445   male    NA     0     0              65306   8.1125                
    ## 446   male  4.00     0     2              33638  81.8583             A34
    ## 447 female 13.00     0     1             250644  19.5000                
    ## 448   male 34.00     0     0             113794  26.5500                
    ## 449 female  5.00     2     1               2666  19.2583                
    ## 450   male 52.00     0     0             113786  30.5000            C104
    ## 451   male 36.00     1     2         C.A. 34651  27.7500                
    ## 452   male    NA     1     0              65303  19.9667                
    ## 453   male 30.00     0     0             113051  27.7500            C111
    ## 454   male 49.00     1     0              17453  89.1042             C92
    ## 455   male    NA     0     0           A/5 2817   8.0500                
    ## 456   male 29.00     0     0             349240   7.8958                
    ## 457   male 65.00     0     0              13509  26.5500             E38
    ## 458 female    NA     1     0              17464  51.8625             D21
    ## 459 female 50.00     0     0       F.C.C. 13531  10.5000                
    ## 460   male    NA     0     0             371060   7.7500                
    ## 461   male 48.00     0     0              19952  26.5500             E12
    ## 462   male 34.00     0     0             364506   8.0500                
    ## 463   male 47.00     0     0             111320  38.5000             E63
    ## 464   male 48.00     0     0             234360  13.0000                
    ## 465   male    NA     0     0           A/S 2816   8.0500                
    ## 466   male 38.00     0     0 SOTON/O.Q. 3101306   7.0500                
    ## 467   male    NA     0     0             239853   0.0000                
    ## 468   male 56.00     0     0             113792  26.5500                
    ## 469   male    NA     0     0              36209   7.7250                
    ## 470 female  0.75     2     1               2666  19.2583                
    ## 471   male    NA     0     0             323592   7.2500                
    ## 472   male 38.00     0     0             315089   8.6625                
    ## 473 female 33.00     1     2         C.A. 34651  27.7500                
    ## 474 female 23.00     0     0    SC/AH Basle 541  13.7917               D
    ## 475 female 22.00     0     0               7553   9.8375                
    ## 476   male    NA     0     0             110465  52.0000             A14
    ## 477   male 34.00     1     0              31027  21.0000                
    ## 478   male 29.00     1     0               3460   7.0458                
    ## 479   male 22.00     0     0             350060   7.5208                
    ## 480 female  2.00     0     1            3101298  12.2875                
    ## 481   male  9.00     5     2            CA 2144  46.9000                
    ## 482   male    NA     0     0             239854   0.0000                
    ## 483   male 50.00     0     0           A/5 3594   8.0500                
    ## 484 female 63.00     0     0               4134   9.5875                
    ## 485   male 25.00     1     0              11967  91.0792             B49
    ## 486 female    NA     3     1               4133  25.4667                
    ## 487 female 35.00     1     0              19943  90.0000             C93
    ## 488   male 58.00     0     0              11771  29.7000             B37
    ## 489   male 30.00     0     0         A.5. 18509   8.0500                
    ## 490   male  9.00     1     1         C.A. 37671  15.9000                
    ## 491   male    NA     1     0              65304  19.9667                
    ## 492   male 21.00     0     0   SOTON/OQ 3101317   7.2500                
    ## 493   male 55.00     0     0             113787  30.5000             C30
    ## 494   male 71.00     0     0           PC 17609  49.5042                
    ## 495   male 21.00     0     0          A/4 45380   8.0500                
    ## 496   male    NA     0     0               2627  14.4583                
    ## 497 female 54.00     1     0              36947  78.2667             D20
    ## 498   male    NA     0     0          C.A. 6212  15.1000                
    ## 499 female 25.00     1     2             113781 151.5500         C22 C26
    ## 500   male 24.00     0     0             350035   7.7958                
    ## 501   male 17.00     0     0             315086   8.6625                
    ## 502 female 21.00     0     0             364846   7.7500                
    ## 503 female    NA     0     0             330909   7.6292                
    ## 504 female 37.00     0     0               4135   9.5875                
    ## 505 female 16.00     0     0             110152  86.5000             B79
    ## 506   male 18.00     1     0           PC 17758 108.9000             C65
    ## 507 female 33.00     0     2              26360  26.0000                
    ## 508   male    NA     0     0             111427  26.5500                
    ## 509   male 28.00     0     0             C 4001  22.5250                
    ## 510   male 26.00     0     0               1601  56.4958                
    ## 511   male 29.00     0     0             382651   7.7500                
    ## 512   male    NA     0     0   SOTON/OQ 3101316   8.0500                
    ## 513   male 36.00     0     0           PC 17473  26.2875             E25
    ## 514 female 54.00     1     0           PC 17603  59.4000                
    ## 515   male 24.00     0     0             349209   7.4958                
    ## 516   male 47.00     0     0              36967  34.0208             D46
    ## 517 female 34.00     0     0         C.A. 34260  10.5000             F33
    ## 518   male    NA     0     0             371110  24.1500                
    ## 519 female 36.00     1     0             226875  26.0000                
    ## 520   male 32.00     0     0             349242   7.8958                
    ## 521 female 30.00     0     0              12749  93.5000             B73
    ## 522   male 22.00     0     0             349252   7.8958                
    ## 523   male    NA     0     0               2624   7.2250                
    ## 524 female 44.00     0     1             111361  57.9792             B18
    ## 525   male    NA     0     0               2700   7.2292                
    ## 526   male 40.50     0     0             367232   7.7500                
    ## 527 female 50.00     0     0        W./C. 14258  10.5000                
    ## 528   male    NA     0     0           PC 17483 221.7792             C95
    ## 529   male 39.00     0     0            3101296   7.9250                
    ## 530   male 23.00     2     1              29104  11.5000                
    ## 531 female  2.00     1     1              26360  26.0000                
    ## 532   male    NA     0     0               2641   7.2292                
    ## 533   male 17.00     1     1               2690   7.2292                
    ## 534 female    NA     0     2               2668  22.3583                
    ## 535 female 30.00     0     0             315084   8.6625                
    ## 536 female  7.00     0     2       F.C.C. 13529  26.2500                
    ## 537   male 45.00     0     0             113050  26.5500             B38
    ## 538 female 30.00     0     0           PC 17761 106.4250                
    ## 539   male    NA     0     0             364498  14.5000                
    ## 540 female 22.00     0     2              13568  49.5000             B39
    ## 541 female 36.00     0     2          WE/P 5735  71.0000             B22
    ## 542 female  9.00     4     2             347082  31.2750                
    ## 543 female 11.00     4     2             347082  31.2750                
    ## 544   male 32.00     1     0               2908  26.0000                
    ## 545   male 50.00     1     0           PC 17761 106.4250             C86
    ## 546   male 64.00     0     0                693  26.0000                
    ## 547 female 19.00     1     0               2908  26.0000                
    ## 548   male    NA     0     0      SC/PARIS 2146  13.8625                
    ## 549   male 33.00     1     1             363291  20.5250                
    ## 550   male  8.00     1     1         C.A. 33112  36.7500                
    ## 551   male 17.00     0     2              17421 110.8833             C70
    ## 552   male 27.00     0     0             244358  26.0000                
    ## 553   male    NA     0     0             330979   7.8292                
    ## 554   male 22.00     0     0               2620   7.2250                
    ## 555 female 22.00     0     0             347085   7.7750                
    ## 556   male 62.00     0     0             113807  26.5500                
    ## 557 female 48.00     1     0              11755  39.6000             A16
    ## 558   male    NA     0     0           PC 17757 227.5250                
    ## 559 female 39.00     1     1             110413  79.6500             E67
    ## 560 female 36.00     1     0             345572  17.4000                
    ## 561   male    NA     0     0             372622   7.7500                
    ## 562   male 40.00     0     0             349251   7.8958                
    ## 563   male 28.00     0     0             218629  13.5000                
    ## 564   male    NA     0     0    SOTON/OQ 392082   8.0500                
    ## 565 female    NA     0     0  SOTON/O.Q. 392087   8.0500                
    ## 566   male 24.00     2     0          A/4 48871  24.1500                
    ## 567   male 19.00     0     0             349205   7.8958                
    ## 568 female 29.00     0     4             349909  21.0750                
    ## 569   male    NA     0     0               2686   7.2292                
    ## 570   male 32.00     0     0             350417   7.8542                
    ## 571   male 62.00     0     0        S.W./PP 752  10.5000                
    ## 572 female 53.00     2     0              11769  51.4792            C101
    ## 573   male 36.00     0     0           PC 17474  26.3875             E25
    ## 574 female    NA     0     0              14312   7.7500                
    ## 575   male 16.00     0     0         A/4. 20589   8.0500                
    ## 576   male 19.00     0     0             358585  14.5000                
    ## 577 female 34.00     0     0             243880  13.0000                
    ## 578 female 39.00     1     0              13507  55.9000             E44
    ## 579 female    NA     1     0               2689  14.4583                
    ## 580   male 32.00     0     0  STON/O 2. 3101286   7.9250                
    ## 581 female 25.00     1     1             237789  30.0000                
    ## 582 female 39.00     1     1              17421 110.8833             C68
    ## 583   male 54.00     0     0              28403  26.0000                
    ## 584   male 36.00     0     0              13049  40.1250             A10
    ## 585   male    NA     0     0               3411   8.7125                
    ## 586 female 18.00     0     2             110413  79.6500             E68
    ## 587   male 47.00     0     0             237565  15.0000                
    ## 588   male 60.00     1     1              13567  79.2000             B41
    ## 589   male 22.00     0     0              14973   8.0500                
    ## 590   male    NA     0     0         A./5. 3235   8.0500                
    ## 591   male 35.00     0     0  STON/O 2. 3101273   7.1250                
    ## 592 female 52.00     1     0              36947  78.2667             D20
    ## 593   male 47.00     0     0           A/5 3902   7.2500                
    ## 594 female    NA     0     2             364848   7.7500                
    ## 595   male 37.00     1     0        SC/AH 29037  26.0000                
    ## 596   male 36.00     1     1             345773  24.1500                
    ## 597 female    NA     0     0             248727  33.0000                
    ## 598   male 49.00     0     0               LINE   0.0000                
    ## 599   male    NA     0     0               2664   7.2250                
    ## 600   male 49.00     1     0           PC 17485  56.9292             A20
    ## 601 female 24.00     2     1             243847  27.0000                
    ## 602   male    NA     0     0             349214   7.8958                
    ## 603   male    NA     0     0             113796  42.4000                
    ## 604   male 44.00     0     0             364511   8.0500                
    ## 605   male 35.00     0     0             111426  26.5500                
    ## 606   male 36.00     1     0             349910  15.5500                
    ## 607   male 30.00     0     0             349246   7.8958                
    ## 608   male 27.00     0     0             113804  30.5000                
    ## 609 female 22.00     1     2      SC/Paris 2123  41.5792                
    ## 610 female 40.00     0     0           PC 17582 153.4625            C125
    ## 611 female 39.00     1     5             347082  31.2750                
    ## 612   male    NA     0     0 SOTON/O.Q. 3101305   7.0500                
    ## 613 female    NA     1     0             367230  15.5000                
    ## 614   male    NA     0     0             370377   7.7500                
    ## 615   male 35.00     0     0             364512   8.0500                
    ## 616 female 24.00     1     2             220845  65.0000                
    ## 617   male 34.00     1     1             347080  14.4000                
    ## 618 female 26.00     1     0          A/5. 3336  16.1000                
    ## 619 female  4.00     2     1             230136  39.0000              F4
    ## 620   male 26.00     0     0              31028  10.5000                
    ## 621   male 27.00     1     0               2659  14.4542                
    ## 622   male 42.00     1     0              11753  52.5542             D19
    ## 623   male 20.00     1     1               2653  15.7417                
    ## 624   male 21.00     0     0             350029   7.8542                
    ## 625   male 21.00     0     0              54636  16.1000                
    ## 626   male 61.00     0     0              36963  32.3208             D50
    ## 627   male 57.00     0     0             219533  12.3500                
    ## 628 female 21.00     0     0              13502  77.9583              D9
    ## 629   male 26.00     0     0             349224   7.8958                
    ## 630   male    NA     0     0             334912   7.7333                
    ## 631   male 80.00     0     0              27042  30.0000             A23
    ## 632   male 51.00     0     0             347743   7.0542                
    ## 633   male 32.00     0     0              13214  30.5000             B50
    ## 634   male    NA     0     0             112052   0.0000                
    ## 635 female  9.00     3     2             347088  27.9000                
    ## 636 female 28.00     0     0             237668  13.0000                
    ## 637   male 32.00     0     0  STON/O 2. 3101292   7.9250                
    ## 638   male 31.00     1     1         C.A. 31921  26.2500                
    ## 639 female 41.00     0     5            3101295  39.6875                
    ## 640   male    NA     1     0             376564  16.1000                
    ## 641   male 20.00     0     0             350050   7.8542                
    ## 642 female 24.00     0     0           PC 17477  69.3000             B35
    ## 643 female  2.00     3     2             347088  27.9000                
    ## 644   male    NA     0     0               1601  56.4958                
    ## 645 female  0.75     2     1               2666  19.2583                
    ## 646   male 48.00     1     0           PC 17572  76.7292             D33
    ## 647   male 19.00     0     0             349231   7.8958                
    ## 648   male 56.00     0     0              13213  35.5000             A26
    ## 649   male    NA     0     0      S.O./P.P. 751   7.5500                
    ## 650 female 23.00     0     0           CA. 2314   7.5500                
    ## 651   male    NA     0     0             349221   7.8958                
    ## 652 female 18.00     0     1             231919  23.0000                
    ## 653   male 21.00     0     0               8475   8.4333                
    ## 654 female    NA     0     0             330919   7.8292                
    ## 655 female 18.00     0     0             365226   6.7500                
    ## 656   male 24.00     2     0       S.O.C. 14879  73.5000                
    ## 657   male    NA     0     0             349223   7.8958                
    ## 658 female 32.00     1     1             364849  15.5000                
    ## 659   male 23.00     0     0              29751  13.0000                
    ## 660   male 58.00     0     2              35273 113.2750             D48
    ## 661   male 50.00     2     0           PC 17611 133.6500                
    ## 662   male 40.00     0     0               2623   7.2250                
    ## 663   male 47.00     0     0               5727  25.5875             E58
    ## 664   male 36.00     0     0             349210   7.4958                
    ## 665   male 20.00     1     0  STON/O 2. 3101285   7.9250                
    ## 666   male 32.00     2     0       S.O.C. 14879  73.5000                
    ## 667   male 25.00     0     0             234686  13.0000                
    ## 668   male    NA     0     0             312993   7.7750                
    ## 669   male 43.00     0     0           A/5 3536   8.0500                
    ## 670 female    NA     1     0              19996  52.0000            C126
    ## 671 female 40.00     1     1              29750  39.0000                
    ## 672   male 31.00     1     0         F.C. 12750  52.0000             B71
    ## 673   male 70.00     0     0         C.A. 24580  10.5000                
    ## 674   male 31.00     0     0             244270  13.0000                
    ## 675   male    NA     0     0             239856   0.0000                
    ## 676   male 18.00     0     0             349912   7.7750                
    ## 677   male 24.50     0     0             342826   8.0500                
    ## 678 female 18.00     0     0               4138   9.8417                
    ## 679 female 43.00     1     6            CA 2144  46.9000                
    ## 680   male 36.00     0     1           PC 17755 512.3292     B51 B53 B55
    ## 681 female    NA     0     0             330935   8.1375                
    ## 682   male 27.00     0     0           PC 17572  76.7292             D49
    ## 683   male 20.00     0     0               6563   9.2250                
    ## 684   male 14.00     5     2            CA 2144  46.9000                
    ## 685   male 60.00     1     1              29750  39.0000                
    ## 686   male 25.00     1     2      SC/Paris 2123  41.5792                
    ## 687   male 14.00     4     1            3101295  39.6875                
    ## 688   male 19.00     0     0             349228  10.1708                
    ## 689   male 18.00     0     0             350036   7.7958                
    ## 690 female 15.00     0     1              24160 211.3375              B5
    ## 691   male 31.00     1     0              17474  57.0000             B20
    ## 692 female  4.00     0     1             349256  13.4167                
    ## 693   male    NA     0     0               1601  56.4958                
    ## 694   male 25.00     0     0               2672   7.2250                
    ## 695   male 60.00     0     0             113800  26.5500                
    ## 696   male 52.00     0     0             248731  13.5000                
    ## 697   male 44.00     0     0             363592   8.0500                
    ## 698 female    NA     0     0              35852   7.7333                
    ## 699   male 49.00     1     1              17421 110.8833             C68
    ## 700   male 42.00     0     0             348121   7.6500           F G63
    ## 701 female 18.00     1     0           PC 17757 227.5250         C62 C64
    ## 702   male 35.00     0     0           PC 17475  26.2875             E24
    ## 703 female 18.00     0     1               2691  14.4542                
    ## 704   male 25.00     0     0              36864   7.7417                
    ## 705   male 26.00     1     0             350025   7.8542                
    ## 706   male 39.00     0     0             250655  26.0000                
    ## 707 female 45.00     0     0             223596  13.5000                
    ## 708   male 42.00     0     0           PC 17476  26.2875             E24
    ## 709 female 22.00     0     0             113781 151.5500                
    ## 710   male    NA     1     1               2661  15.2458                
    ## 711 female 24.00     0     0           PC 17482  49.5042             C90
    ## 712   male    NA     0     0             113028  26.5500            C124
    ## 713   male 48.00     1     0              19996  52.0000            C126
    ## 714   male 29.00     0     0               7545   9.4833                
    ## 715   male 52.00     0     0             250647  13.0000                
    ## 716   male 19.00     0     0             348124   7.6500           F G73
    ## 717 female 38.00     0     0           PC 17757 227.5250             C45
    ## 718 female 27.00     0     0              34218  10.5000            E101
    ## 719   male    NA     0     0              36568  15.5000                
    ## 720   male 33.00     0     0             347062   7.7750                
    ## 721 female  6.00     0     1             248727  33.0000                
    ## 722   male 17.00     1     0             350048   7.0542                
    ## 723   male 34.00     0     0              12233  13.0000                
    ## 724   male 50.00     0     0             250643  13.0000                
    ## 725   male 27.00     1     0             113806  53.1000              E8
    ## 726   male 20.00     0     0             315094   8.6625                
    ## 727 female 30.00     3     0              31027  21.0000                
    ## 728 female    NA     0     0              36866   7.7375                
    ## 729   male 25.00     1     0             236853  26.0000                
    ## 730 female 25.00     1     0   STON/O2. 3101271   7.9250                
    ## 731 female 29.00     0     0              24160 211.3375              B5
    ## 732   male 11.00     0     0               2699  18.7875                
    ## 733   male    NA     0     0             239855   0.0000                
    ## 734   male 23.00     0     0              28425  13.0000                
    ## 735   male 23.00     0     0             233639  13.0000                
    ## 736   male 28.50     0     0              54636  16.1000                
    ## 737 female 48.00     1     3         W./C. 6608  34.3750                
    ## 738   male 35.00     0     0           PC 17755 512.3292            B101
    ## 739   male    NA     0     0             349201   7.8958                
    ## 740   male    NA     0     0             349218   7.8958                
    ## 741   male    NA     0     0              16988  30.0000             D45
    ## 742   male 36.00     1     0              19877  78.8500             C46
    ## 743 female 21.00     2     2           PC 17608 262.3750 B57 B59 B63 B66
    ## 744   male 24.00     1     0             376566  16.1000                
    ## 745   male 31.00     0     0  STON/O 2. 3101288   7.9250                
    ## 746   male 70.00     1     1          WE/P 5735  71.0000             B22
    ## 747   male 16.00     1     1          C.A. 2673  20.2500                
    ## 748 female 30.00     0     0             250648  13.0000                
    ## 749   male 19.00     1     0             113773  53.1000             D30
    ## 750   male 31.00     0     0             335097   7.7500                
    ## 751 female  4.00     1     1              29103  23.0000                
    ## 752   male  6.00     0     1             392096  12.4750            E121
    ## 753   male 33.00     0     0             345780   9.5000                
    ## 754   male 23.00     0     0             349204   7.8958                
    ## 755 female 48.00     1     2             220845  65.0000                
    ## 756   male  0.67     1     1             250649  14.5000                
    ## 757   male 28.00     0     0             350042   7.7958                
    ## 758   male 18.00     0     0              29108  11.5000                
    ## 759   male 34.00     0     0             363294   8.0500                
    ## 760 female 33.00     0     0             110152  86.5000             B77
    ## 761   male    NA     0     0             358585  14.5000                
    ## 762   male 41.00     0     0   SOTON/O2 3101272   7.1250                
    ## 763   male 20.00     0     0               2663   7.2292                
    ## 764 female 36.00     1     2             113760 120.0000         B96 B98
    ## 765   male 16.00     0     0             347074   7.7750                
    ## 766 female 51.00     1     0              13502  77.9583             D11
    ## 767   male    NA     0     0             112379  39.6000                
    ## 768 female 30.50     0     0             364850   7.7500                
    ## 769   male    NA     1     0             371110  24.1500                
    ## 770   male 32.00     0     0               8471   8.3625                
    ## 771   male 24.00     0     0             345781   9.5000                
    ## 772   male 48.00     0     0             350047   7.8542                
    ## 773 female 57.00     0     0        S.O./P.P. 3  10.5000             E77
    ## 774   male    NA     0     0               2674   7.2250                
    ## 775 female 54.00     1     3              29105  23.0000                
    ## 776   male 18.00     0     0             347078   7.7500                
    ## 777   male    NA     0     0             383121   7.7500             F38
    ## 778 female  5.00     0     0             364516  12.4750                
    ## 779   male    NA     0     0              36865   7.7375                
    ## 780 female 43.00     0     1              24160 211.3375              B3
    ## 781 female 13.00     0     0               2687   7.2292                
    ## 782 female 17.00     1     0              17474  57.0000             B20
    ## 783   male 29.00     0     0             113501  30.0000              D6
    ## 784   male    NA     1     2         W./C. 6607  23.4500                
    ## 785   male 25.00     0     0 SOTON/O.Q. 3101312   7.0500                
    ## 786   male 25.00     0     0             374887   7.2500                
    ## 787 female 18.00     0     0            3101265   7.4958                
    ## 788   male  8.00     4     1             382652  29.1250                
    ## 789   male  1.00     1     2          C.A. 2315  20.5750                
    ## 790   male 46.00     0     0           PC 17593  79.2000         B82 B84
    ## 791   male    NA     0     0              12460   7.7500                
    ## 792   male 16.00     0     0             239865  26.0000                
    ## 793 female    NA     8     2           CA. 2343  69.5500                
    ## 794   male    NA     0     0           PC 17600  30.6958                
    ## 795   male 25.00     0     0             349203   7.8958                
    ## 796   male 39.00     0     0              28213  13.0000                
    ## 797 female 49.00     0     0              17465  25.9292             D17
    ## 798 female 31.00     0     0             349244   8.6833                
    ## 799   male 30.00     0     0               2685   7.2292                
    ## 800 female 30.00     1     1             345773  24.1500                
    ## 801   male 34.00     0     0             250647  13.0000                
    ## 802 female 31.00     1     1         C.A. 31921  26.2500                
    ## 803   male 11.00     1     2             113760 120.0000         B96 B98
    ## 804   male  0.42     0     1               2625   8.5167                
    ## 805   male 27.00     0     0             347089   6.9750                
    ## 806   male 31.00     0     0             347063   7.7750                
    ## 807   male 39.00     0     0             112050   0.0000             A36
    ## 808 female 18.00     0     0             347087   7.7750                
    ## 809   male 39.00     0     0             248723  13.0000                
    ## 810 female 33.00     1     0             113806  53.1000              E8
    ## 811   male 26.00     0     0               3474   7.8875                
    ## 812   male 39.00     0     0          A/4 48871  24.1500                
    ## 813   male 35.00     0     0              28206  10.5000                
    ## 814 female  6.00     4     2             347082  31.2750                
    ## 815   male 30.50     0     0             364499   8.0500                
    ## 816   male    NA     0     0             112058   0.0000            B102
    ## 817 female 23.00     0     0   STON/O2. 3101290   7.9250                
    ## 818   male 31.00     1     1    S.C./PARIS 2079  37.0042                
    ## 819   male 43.00     0     0             C 7075   6.4500                
    ## 820   male 10.00     3     2             347088  27.9000                
    ## 821 female 52.00     1     1              12749  93.5000             B69
    ## 822   male 27.00     0     0             315098   8.6625                
    ## 823   male 38.00     0     0              19972   0.0000                
    ## 824 female 27.00     0     1             392096  12.4750            E121
    ## 825   male  2.00     4     1            3101295  39.6875                
    ## 826   male    NA     0     0             368323   6.9500                
    ## 827   male    NA     0     0               1601  56.4958                
    ## 828   male  1.00     0     2    S.C./PARIS 2079  37.0042                
    ## 829   male    NA     0     0             367228   7.7500                
    ## 830 female 62.00     0     0             113572  80.0000             B28
    ## 831 female 15.00     1     0               2659  14.4542                
    ## 832   male  0.83     1     1              29106  18.7500                
    ## 833   male    NA     0     0               2671   7.2292                
    ## 834   male 23.00     0     0             347468   7.8542                
    ## 835   male 18.00     0     0               2223   8.3000                
    ## 836 female 39.00     1     1           PC 17756  83.1583             E49
    ## 837   male 21.00     0     0             315097   8.6625                
    ## 838   male    NA     0     0             392092   8.0500                
    ## 839   male 32.00     0     0               1601  56.4958                
    ## 840   male    NA     0     0              11774  29.7000             C47
    ## 841   male 20.00     0     0   SOTON/O2 3101287   7.9250                
    ## 842   male 16.00     0     0        S.O./P.P. 3  10.5000                
    ## 843 female 30.00     0     0             113798  31.0000                
    ## 844   male 34.50     0     0               2683   6.4375                
    ## 845   male 17.00     0     0             315090   8.6625                
    ## 846   male 42.00     0     0          C.A. 5547   7.5500                
    ## 847   male    NA     8     2           CA. 2343  69.5500                
    ## 848   male 35.00     0     0             349213   7.8958                
    ## 849   male 28.00     0     1             248727  33.0000                
    ## 850 female    NA     1     0              17453  89.1042             C92
    ## 851   male  4.00     4     2             347082  31.2750                
    ## 852   male 74.00     0     0             347060   7.7750                
    ## 853 female  9.00     1     1               2678  15.2458                
    ## 854 female 16.00     0     1           PC 17592  39.4000             D28
    ## 855 female 44.00     1     0             244252  26.0000                
    ## 856 female 18.00     0     1             392091   9.3500                
    ## 857 female 45.00     1     1              36928 164.8667                
    ## 858   male 51.00     0     0             113055  26.5500             E17
    ## 859 female 24.00     0     3               2666  19.2583                
    ## 860   male    NA     0     0               2629   7.2292                
    ## 861   male 41.00     2     0             350026  14.1083                
    ## 862   male 21.00     1     0              28134  11.5000                
    ## 863 female 48.00     0     0              17466  25.9292             D17
    ## 864 female    NA     8     2           CA. 2343  69.5500                
    ## 865   male 24.00     0     0             233866  13.0000                
    ## 866 female 42.00     0     0             236852  13.0000                
    ## 867 female 27.00     1     0      SC/PARIS 2149  13.8583                
    ## 868   male 31.00     0     0           PC 17590  50.4958             A24
    ## 869   male    NA     0     0             345777   9.5000                
    ## 870   male  4.00     1     1             347742  11.1333                
    ## 871   male 26.00     0     0             349248   7.8958                
    ## 872 female 47.00     1     1              11751  52.5542             D35
    ## 873   male 33.00     0     0                695   5.0000     B51 B53 B55
    ## 874   male 47.00     0     0             345765   9.0000                
    ## 875 female 28.00     1     0          P/PP 3381  24.0000                
    ## 876 female 15.00     0     0               2667   7.2250                
    ## 877   male 20.00     0     0               7534   9.8458                
    ## 878   male 19.00     0     0             349212   7.8958                
    ## 879   male    NA     0     0             349217   7.8958                
    ## 880 female 56.00     0     1              11767  83.1583             C50
    ## 881 female 25.00     0     1             230433  26.0000                
    ## 882   male 33.00     0     0             349257   7.8958                
    ## 883 female 22.00     0     0               7552  10.5167                
    ## 884   male 28.00     0     0   C.A./SOTON 34068  10.5000                
    ## 885   male 25.00     0     0    SOTON/OQ 392076   7.0500                
    ## 886 female 39.00     0     5             382652  29.1250                
    ## 887   male 27.00     0     0             211536  13.0000                
    ## 888 female 19.00     0     0             112053  30.0000             B42
    ## 889 female    NA     1     2         W./C. 6607  23.4500                
    ## 890   male 26.00     0     0             111369  30.0000            C148
    ## 891   male 32.00     0     0             370376   7.7500                
    ##     Embarked
    ## 1          S
    ## 2          C
    ## 3          S
    ## 4          S
    ## 5          S
    ## 6          Q
    ## 7          S
    ## 8          S
    ## 9          S
    ## 10         C
    ## 11         S
    ## 12         S
    ## 13         S
    ## 14         S
    ## 15         S
    ## 16         S
    ## 17         Q
    ## 18         S
    ## 19         S
    ## 20         C
    ## 21         S
    ## 22         S
    ## 23         Q
    ## 24         S
    ## 25         S
    ## 26         S
    ## 27         C
    ## 28         S
    ## 29         Q
    ## 30         S
    ## 31         C
    ## 32         C
    ## 33         Q
    ## 34         S
    ## 35         C
    ## 36         S
    ## 37         C
    ## 38         S
    ## 39         S
    ## 40         C
    ## 41         S
    ## 42         S
    ## 43         C
    ## 44         C
    ## 45         Q
    ## 46         S
    ## 47         Q
    ## 48         Q
    ## 49         C
    ## 50         S
    ## 51         S
    ## 52         S
    ## 53         C
    ## 54         S
    ## 55         C
    ## 56         S
    ## 57         S
    ## 58         C
    ## 59         S
    ## 60         S
    ## 61         C
    ## 62          
    ## 63         S
    ## 64         S
    ## 65         C
    ## 66         C
    ## 67         S
    ## 68         S
    ## 69         S
    ## 70         S
    ## 71         S
    ## 72         S
    ## 73         S
    ## 74         C
    ## 75         S
    ## 76         S
    ## 77         S
    ## 78         S
    ## 79         S
    ## 80         S
    ## 81         S
    ## 82         S
    ## 83         Q
    ## 84         S
    ## 85         S
    ## 86         S
    ## 87         S
    ## 88         S
    ## 89         S
    ## 90         S
    ## 91         S
    ## 92         S
    ## 93         S
    ## 94         S
    ## 95         S
    ## 96         S
    ## 97         C
    ## 98         C
    ## 99         S
    ## 100        S
    ## 101        S
    ## 102        S
    ## 103        S
    ## 104        S
    ## 105        S
    ## 106        S
    ## 107        S
    ## 108        S
    ## 109        S
    ## 110        Q
    ## 111        S
    ## 112        C
    ## 113        S
    ## 114        S
    ## 115        C
    ## 116        S
    ## 117        Q
    ## 118        S
    ## 119        C
    ## 120        S
    ## 121        S
    ## 122        S
    ## 123        C
    ## 124        S
    ## 125        S
    ## 126        C
    ## 127        Q
    ## 128        S
    ## 129        C
    ## 130        S
    ## 131        C
    ## 132        S
    ## 133        S
    ## 134        S
    ## 135        S
    ## 136        C
    ## 137        S
    ## 138        S
    ## 139        S
    ## 140        C
    ## 141        C
    ## 142        S
    ## 143        S
    ## 144        Q
    ## 145        S
    ## 146        S
    ## 147        S
    ## 148        S
    ## 149        S
    ## 150        S
    ## 151        S
    ## 152        S
    ## 153        S
    ## 154        S
    ## 155        S
    ## 156        C
    ## 157        Q
    ## 158        S
    ## 159        S
    ## 160        S
    ## 161        S
    ## 162        S
    ## 163        S
    ## 164        S
    ## 165        S
    ## 166        S
    ## 167        S
    ## 168        S
    ## 169        S
    ## 170        S
    ## 171        S
    ## 172        Q
    ## 173        S
    ## 174        S
    ## 175        C
    ## 176        S
    ## 177        S
    ## 178        C
    ## 179        S
    ## 180        S
    ## 181        S
    ## 182        C
    ## 183        S
    ## 184        S
    ## 185        S
    ## 186        S
    ## 187        Q
    ## 188        S
    ## 189        Q
    ## 190        S
    ## 191        S
    ## 192        S
    ## 193        S
    ## 194        S
    ## 195        C
    ## 196        C
    ## 197        Q
    ## 198        S
    ## 199        Q
    ## 200        S
    ## 201        S
    ## 202        S
    ## 203        S
    ## 204        C
    ## 205        S
    ## 206        S
    ## 207        S
    ## 208        C
    ## 209        Q
    ## 210        C
    ## 211        S
    ## 212        S
    ## 213        S
    ## 214        S
    ## 215        Q
    ## 216        C
    ## 217        S
    ## 218        S
    ## 219        C
    ## 220        S
    ## 221        S
    ## 222        S
    ## 223        S
    ## 224        S
    ## 225        S
    ## 226        S
    ## 227        S
    ## 228        S
    ## 229        S
    ## 230        S
    ## 231        S
    ## 232        S
    ## 233        S
    ## 234        S
    ## 235        S
    ## 236        S
    ## 237        S
    ## 238        S
    ## 239        S
    ## 240        S
    ## 241        C
    ## 242        Q
    ## 243        S
    ## 244        S
    ## 245        C
    ## 246        Q
    ## 247        S
    ## 248        S
    ## 249        S
    ## 250        S
    ## 251        S
    ## 252        S
    ## 253        S
    ## 254        S
    ## 255        S
    ## 256        C
    ## 257        C
    ## 258        S
    ## 259        C
    ## 260        S
    ## 261        Q
    ## 262        S
    ## 263        S
    ## 264        S
    ## 265        Q
    ## 266        S
    ## 267        S
    ## 268        S
    ## 269        S
    ## 270        S
    ## 271        S
    ## 272        S
    ## 273        S
    ## 274        C
    ## 275        Q
    ## 276        S
    ## 277        S
    ## 278        S
    ## 279        Q
    ## 280        S
    ## 281        Q
    ## 282        S
    ## 283        S
    ## 284        S
    ## 285        S
    ## 286        C
    ## 287        S
    ## 288        S
    ## 289        S
    ## 290        Q
    ## 291        S
    ## 292        C
    ## 293        C
    ## 294        S
    ## 295        S
    ## 296        C
    ## 297        C
    ## 298        S
    ## 299        S
    ## 300        C
    ## 301        Q
    ## 302        Q
    ## 303        S
    ## 304        Q
    ## 305        S
    ## 306        S
    ## 307        C
    ## 308        C
    ## 309        C
    ## 310        C
    ## 311        C
    ## 312        C
    ## 313        S
    ## 314        S
    ## 315        S
    ## 316        S
    ## 317        S
    ## 318        S
    ## 319        S
    ## 320        C
    ## 321        S
    ## 322        S
    ## 323        Q
    ## 324        S
    ## 325        S
    ## 326        C
    ## 327        S
    ## 328        S
    ## 329        S
    ## 330        C
    ## 331        Q
    ## 332        S
    ## 333        S
    ## 334        S
    ## 335        S
    ## 336        S
    ## 337        S
    ## 338        C
    ## 339        S
    ## 340        S
    ## 341        S
    ## 342        S
    ## 343        S
    ## 344        S
    ## 345        S
    ## 346        S
    ## 347        S
    ## 348        S
    ## 349        S
    ## 350        S
    ## 351        S
    ## 352        S
    ## 353        C
    ## 354        S
    ## 355        C
    ## 356        S
    ## 357        S
    ## 358        S
    ## 359        Q
    ## 360        Q
    ## 361        S
    ## 362        C
    ## 363        C
    ## 364        S
    ## 365        Q
    ## 366        S
    ## 367        C
    ## 368        C
    ## 369        Q
    ## 370        C
    ## 371        C
    ## 372        S
    ## 373        S
    ## 374        C
    ## 375        S
    ## 376        C
    ## 377        S
    ## 378        C
    ## 379        C
    ## 380        S
    ## 381        C
    ## 382        C
    ## 383        S
    ## 384        S
    ## 385        S
    ## 386        S
    ## 387        S
    ## 388        S
    ## 389        Q
    ## 390        C
    ## 391        S
    ## 392        S
    ## 393        S
    ## 394        C
    ## 395        S
    ## 396        S
    ## 397        S
    ## 398        S
    ## 399        S
    ## 400        S
    ## 401        S
    ## 402        S
    ## 403        S
    ## 404        S
    ## 405        S
    ## 406        S
    ## 407        S
    ## 408        S
    ## 409        S
    ## 410        S
    ## 411        S
    ## 412        Q
    ## 413        Q
    ## 414        S
    ## 415        S
    ## 416        S
    ## 417        S
    ## 418        S
    ## 419        S
    ## 420        S
    ## 421        C
    ## 422        Q
    ## 423        S
    ## 424        S
    ## 425        S
    ## 426        S
    ## 427        S
    ## 428        S
    ## 429        Q
    ## 430        S
    ## 431        S
    ## 432        S
    ## 433        S
    ## 434        S
    ## 435        S
    ## 436        S
    ## 437        S
    ## 438        S
    ## 439        S
    ## 440        S
    ## 441        S
    ## 442        S
    ## 443        S
    ## 444        S
    ## 445        S
    ## 446        S
    ## 447        S
    ## 448        S
    ## 449        C
    ## 450        S
    ## 451        S
    ## 452        S
    ## 453        C
    ## 454        C
    ## 455        S
    ## 456        C
    ## 457        S
    ## 458        S
    ## 459        S
    ## 460        Q
    ## 461        S
    ## 462        S
    ## 463        S
    ## 464        S
    ## 465        S
    ## 466        S
    ## 467        S
    ## 468        S
    ## 469        Q
    ## 470        C
    ## 471        S
    ## 472        S
    ## 473        S
    ## 474        C
    ## 475        S
    ## 476        S
    ## 477        S
    ## 478        S
    ## 479        S
    ## 480        S
    ## 481        S
    ## 482        S
    ## 483        S
    ## 484        S
    ## 485        C
    ## 486        S
    ## 487        S
    ## 488        C
    ## 489        S
    ## 490        S
    ## 491        S
    ## 492        S
    ## 493        S
    ## 494        C
    ## 495        S
    ## 496        C
    ## 497        C
    ## 498        S
    ## 499        S
    ## 500        S
    ## 501        S
    ## 502        Q
    ## 503        Q
    ## 504        S
    ## 505        S
    ## 506        C
    ## 507        S
    ## 508        S
    ## 509        S
    ## 510        S
    ## 511        Q
    ## 512        S
    ## 513        S
    ## 514        C
    ## 515        S
    ## 516        S
    ## 517        S
    ## 518        Q
    ## 519        S
    ## 520        S
    ## 521        S
    ## 522        S
    ## 523        C
    ## 524        C
    ## 525        C
    ## 526        Q
    ## 527        S
    ## 528        S
    ## 529        S
    ## 530        S
    ## 531        S
    ## 532        C
    ## 533        C
    ## 534        C
    ## 535        S
    ## 536        S
    ## 537        S
    ## 538        C
    ## 539        S
    ## 540        C
    ## 541        S
    ## 542        S
    ## 543        S
    ## 544        S
    ## 545        C
    ## 546        S
    ## 547        S
    ## 548        C
    ## 549        S
    ## 550        S
    ## 551        C
    ## 552        S
    ## 553        Q
    ## 554        C
    ## 555        S
    ## 556        S
    ## 557        C
    ## 558        C
    ## 559        S
    ## 560        S
    ## 561        Q
    ## 562        S
    ## 563        S
    ## 564        S
    ## 565        S
    ## 566        S
    ## 567        S
    ## 568        S
    ## 569        C
    ## 570        S
    ## 571        S
    ## 572        S
    ## 573        S
    ## 574        Q
    ## 575        S
    ## 576        S
    ## 577        S
    ## 578        S
    ## 579        C
    ## 580        S
    ## 581        S
    ## 582        C
    ## 583        S
    ## 584        C
    ## 585        C
    ## 586        S
    ## 587        S
    ## 588        C
    ## 589        S
    ## 590        S
    ## 591        S
    ## 592        C
    ## 593        S
    ## 594        Q
    ## 595        S
    ## 596        S
    ## 597        S
    ## 598        S
    ## 599        C
    ## 600        C
    ## 601        S
    ## 602        S
    ## 603        S
    ## 604        S
    ## 605        C
    ## 606        S
    ## 607        S
    ## 608        S
    ## 609        C
    ## 610        S
    ## 611        S
    ## 612        S
    ## 613        Q
    ## 614        Q
    ## 615        S
    ## 616        S
    ## 617        S
    ## 618        S
    ## 619        S
    ## 620        S
    ## 621        C
    ## 622        S
    ## 623        C
    ## 624        S
    ## 625        S
    ## 626        S
    ## 627        Q
    ## 628        S
    ## 629        S
    ## 630        Q
    ## 631        S
    ## 632        S
    ## 633        C
    ## 634        S
    ## 635        S
    ## 636        S
    ## 637        S
    ## 638        S
    ## 639        S
    ## 640        S
    ## 641        S
    ## 642        C
    ## 643        S
    ## 644        S
    ## 645        C
    ## 646        C
    ## 647        S
    ## 648        C
    ## 649        S
    ## 650        S
    ## 651        S
    ## 652        S
    ## 653        S
    ## 654        Q
    ## 655        Q
    ## 656        S
    ## 657        S
    ## 658        Q
    ## 659        S
    ## 660        C
    ## 661        S
    ## 662        C
    ## 663        S
    ## 664        S
    ## 665        S
    ## 666        S
    ## 667        S
    ## 668        S
    ## 669        S
    ## 670        S
    ## 671        S
    ## 672        S
    ## 673        S
    ## 674        S
    ## 675        S
    ## 676        S
    ## 677        S
    ## 678        S
    ## 679        S
    ## 680        C
    ## 681        Q
    ## 682        C
    ## 683        S
    ## 684        S
    ## 685        S
    ## 686        C
    ## 687        S
    ## 688        S
    ## 689        S
    ## 690        S
    ## 691        S
    ## 692        C
    ## 693        S
    ## 694        C
    ## 695        S
    ## 696        S
    ## 697        S
    ## 698        Q
    ## 699        C
    ## 700        S
    ## 701        C
    ## 702        S
    ## 703        C
    ## 704        Q
    ## 705        S
    ## 706        S
    ## 707        S
    ## 708        S
    ## 709        S
    ## 710        C
    ## 711        C
    ## 712        S
    ## 713        S
    ## 714        S
    ## 715        S
    ## 716        S
    ## 717        C
    ## 718        S
    ## 719        Q
    ## 720        S
    ## 721        S
    ## 722        S
    ## 723        S
    ## 724        S
    ## 725        S
    ## 726        S
    ## 727        S
    ## 728        Q
    ## 729        S
    ## 730        S
    ## 731        S
    ## 732        C
    ## 733        S
    ## 734        S
    ## 735        S
    ## 736        S
    ## 737        S
    ## 738        C
    ## 739        S
    ## 740        S
    ## 741        S
    ## 742        S
    ## 743        C
    ## 744        S
    ## 745        S
    ## 746        S
    ## 747        S
    ## 748        S
    ## 749        S
    ## 750        Q
    ## 751        S
    ## 752        S
    ## 753        S
    ## 754        S
    ## 755        S
    ## 756        S
    ## 757        S
    ## 758        S
    ## 759        S
    ## 760        S
    ## 761        S
    ## 762        S
    ## 763        C
    ## 764        S
    ## 765        S
    ## 766        S
    ## 767        C
    ## 768        Q
    ## 769        Q
    ## 770        S
    ## 771        S
    ## 772        S
    ## 773        S
    ## 774        C
    ## 775        S
    ## 776        S
    ## 777        Q
    ## 778        S
    ## 779        Q
    ## 780        S
    ## 781        C
    ## 782        S
    ## 783        S
    ## 784        S
    ## 785        S
    ## 786        S
    ## 787        S
    ## 788        Q
    ## 789        S
    ## 790        C
    ## 791        Q
    ## 792        S
    ## 793        S
    ## 794        C
    ## 795        S
    ## 796        S
    ## 797        S
    ## 798        S
    ## 799        C
    ## 800        S
    ## 801        S
    ## 802        S
    ## 803        S
    ## 804        C
    ## 805        S
    ## 806        S
    ## 807        S
    ## 808        S
    ## 809        S
    ## 810        S
    ## 811        S
    ## 812        S
    ## 813        S
    ## 814        S
    ## 815        S
    ## 816        S
    ## 817        S
    ## 818        C
    ## 819        S
    ## 820        S
    ## 821        S
    ## 822        S
    ## 823        S
    ## 824        S
    ## 825        S
    ## 826        Q
    ## 827        S
    ## 828        C
    ## 829        Q
    ## 830         
    ## 831        C
    ## 832        S
    ## 833        C
    ## 834        S
    ## 835        S
    ## 836        C
    ## 837        S
    ## 838        S
    ## 839        S
    ## 840        C
    ## 841        S
    ## 842        S
    ## 843        C
    ## 844        C
    ## 845        S
    ## 846        S
    ## 847        S
    ## 848        C
    ## 849        S
    ## 850        C
    ## 851        S
    ## 852        S
    ## 853        C
    ## 854        S
    ## 855        S
    ## 856        S
    ## 857        S
    ## 858        S
    ## 859        C
    ## 860        C
    ## 861        S
    ## 862        S
    ## 863        S
    ## 864        S
    ## 865        S
    ## 866        S
    ## 867        C
    ## 868        S
    ## 869        S
    ## 870        S
    ## 871        S
    ## 872        S
    ## 873        S
    ## 874        S
    ## 875        C
    ## 876        C
    ## 877        S
    ## 878        S
    ## 879        S
    ## 880        C
    ## 881        S
    ## 882        S
    ## 883        S
    ## 884        S
    ## 885        S
    ## 886        Q
    ## 887        S
    ## 888        S
    ## 889        S
    ## 890        C
    ## 891        Q

``` r
#make data ready for rf
titanic_rf_data <- titanic_tree_data %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  na.omit()
titanic_rf_data
```

    ##     Survived Pclass    Sex   Age SibSp Parch     Fare Embarked
    ## 1       Died      3   male 22.00     1     0   7.2500        S
    ## 2   Survived      1 female 38.00     1     0  71.2833        C
    ## 3   Survived      3 female 26.00     0     0   7.9250        S
    ## 4   Survived      1 female 35.00     1     0  53.1000        S
    ## 5       Died      3   male 35.00     0     0   8.0500        S
    ## 7       Died      1   male 54.00     0     0  51.8625        S
    ## 8       Died      3   male  2.00     3     1  21.0750        S
    ## 9   Survived      3 female 27.00     0     2  11.1333        S
    ## 10  Survived      2 female 14.00     1     0  30.0708        C
    ## 11  Survived      3 female  4.00     1     1  16.7000        S
    ## 12  Survived      1 female 58.00     0     0  26.5500        S
    ## 13      Died      3   male 20.00     0     0   8.0500        S
    ## 14      Died      3   male 39.00     1     5  31.2750        S
    ## 15      Died      3 female 14.00     0     0   7.8542        S
    ## 16  Survived      2 female 55.00     0     0  16.0000        S
    ## 17      Died      3   male  2.00     4     1  29.1250        Q
    ## 19      Died      3 female 31.00     1     0  18.0000        S
    ## 21      Died      2   male 35.00     0     0  26.0000        S
    ## 22  Survived      2   male 34.00     0     0  13.0000        S
    ## 23  Survived      3 female 15.00     0     0   8.0292        Q
    ## 24  Survived      1   male 28.00     0     0  35.5000        S
    ## 25      Died      3 female  8.00     3     1  21.0750        S
    ## 26  Survived      3 female 38.00     1     5  31.3875        S
    ## 28      Died      1   male 19.00     3     2 263.0000        S
    ## 31      Died      1   male 40.00     0     0  27.7208        C
    ## 34      Died      2   male 66.00     0     0  10.5000        S
    ## 35      Died      1   male 28.00     1     0  82.1708        C
    ## 36      Died      1   male 42.00     1     0  52.0000        S
    ## 38      Died      3   male 21.00     0     0   8.0500        S
    ## 39      Died      3 female 18.00     2     0  18.0000        S
    ## 40  Survived      3 female 14.00     1     0  11.2417        C
    ## 41      Died      3 female 40.00     1     0   9.4750        S
    ## 42      Died      2 female 27.00     1     0  21.0000        S
    ## 44  Survived      2 female  3.00     1     2  41.5792        C
    ## 45  Survived      3 female 19.00     0     0   7.8792        Q
    ## 50      Died      3 female 18.00     1     0  17.8000        S
    ## 51      Died      3   male  7.00     4     1  39.6875        S
    ## 52      Died      3   male 21.00     0     0   7.8000        S
    ## 53  Survived      1 female 49.00     1     0  76.7292        C
    ## 54  Survived      2 female 29.00     1     0  26.0000        S
    ## 55      Died      1   male 65.00     0     1  61.9792        C
    ## 57  Survived      2 female 21.00     0     0  10.5000        S
    ## 58      Died      3   male 28.50     0     0   7.2292        C
    ## 59  Survived      2 female  5.00     1     2  27.7500        S
    ## 60      Died      3   male 11.00     5     2  46.9000        S
    ## 61      Died      3   male 22.00     0     0   7.2292        C
    ## 62  Survived      1 female 38.00     0     0  80.0000         
    ## 63      Died      1   male 45.00     1     0  83.4750        S
    ## 64      Died      3   male  4.00     3     2  27.9000        S
    ## 67  Survived      2 female 29.00     0     0  10.5000        S
    ## 68      Died      3   male 19.00     0     0   8.1583        S
    ## 69  Survived      3 female 17.00     4     2   7.9250        S
    ## 70      Died      3   male 26.00     2     0   8.6625        S
    ## 71      Died      2   male 32.00     0     0  10.5000        S
    ## 72      Died      3 female 16.00     5     2  46.9000        S
    ## 73      Died      2   male 21.00     0     0  73.5000        S
    ## 74      Died      3   male 26.00     1     0  14.4542        C
    ## 75  Survived      3   male 32.00     0     0  56.4958        S
    ## 76      Died      3   male 25.00     0     0   7.6500        S
    ## 79  Survived      2   male  0.83     0     2  29.0000        S
    ## 80  Survived      3 female 30.00     0     0  12.4750        S
    ## 81      Died      3   male 22.00     0     0   9.0000        S
    ## 82  Survived      3   male 29.00     0     0   9.5000        S
    ## 84      Died      1   male 28.00     0     0  47.1000        S
    ## 85  Survived      2 female 17.00     0     0  10.5000        S
    ## 86  Survived      3 female 33.00     3     0  15.8500        S
    ## 87      Died      3   male 16.00     1     3  34.3750        S
    ## 89  Survived      1 female 23.00     3     2 263.0000        S
    ## 90      Died      3   male 24.00     0     0   8.0500        S
    ## 91      Died      3   male 29.00     0     0   8.0500        S
    ## 92      Died      3   male 20.00     0     0   7.8542        S
    ## 93      Died      1   male 46.00     1     0  61.1750        S
    ## 94      Died      3   male 26.00     1     2  20.5750        S
    ## 95      Died      3   male 59.00     0     0   7.2500        S
    ## 97      Died      1   male 71.00     0     0  34.6542        C
    ## 98  Survived      1   male 23.00     0     1  63.3583        C
    ## 99  Survived      2 female 34.00     0     1  23.0000        S
    ## 100     Died      2   male 34.00     1     0  26.0000        S
    ## 101     Died      3 female 28.00     0     0   7.8958        S
    ## 103     Died      1   male 21.00     0     1  77.2875        S
    ## 104     Died      3   male 33.00     0     0   8.6542        S
    ## 105     Died      3   male 37.00     2     0   7.9250        S
    ## 106     Died      3   male 28.00     0     0   7.8958        S
    ## 107 Survived      3 female 21.00     0     0   7.6500        S
    ## 109     Died      3   male 38.00     0     0   7.8958        S
    ## 111     Died      1   male 47.00     0     0  52.0000        S
    ## 112     Died      3 female 14.50     1     0  14.4542        C
    ## 113     Died      3   male 22.00     0     0   8.0500        S
    ## 114     Died      3 female 20.00     1     0   9.8250        S
    ## 115     Died      3 female 17.00     0     0  14.4583        C
    ## 116     Died      3   male 21.00     0     0   7.9250        S
    ## 117     Died      3   male 70.50     0     0   7.7500        Q
    ## 118     Died      2   male 29.00     1     0  21.0000        S
    ## 119     Died      1   male 24.00     0     1 247.5208        C
    ## 120     Died      3 female  2.00     4     2  31.2750        S
    ## 121     Died      2   male 21.00     2     0  73.5000        S
    ## 123     Died      2   male 32.50     1     0  30.0708        C
    ## 124 Survived      2 female 32.50     0     0  13.0000        S
    ## 125     Died      1   male 54.00     0     1  77.2875        S
    ## 126 Survived      3   male 12.00     1     0  11.2417        C
    ## 128 Survived      3   male 24.00     0     0   7.1417        S
    ## 130     Died      3   male 45.00     0     0   6.9750        S
    ## 131     Died      3   male 33.00     0     0   7.8958        C
    ## 132     Died      3   male 20.00     0     0   7.0500        S
    ## 133     Died      3 female 47.00     1     0  14.5000        S
    ## 134 Survived      2 female 29.00     1     0  26.0000        S
    ## 135     Died      2   male 25.00     0     0  13.0000        S
    ## 136     Died      2   male 23.00     0     0  15.0458        C
    ## 137 Survived      1 female 19.00     0     2  26.2833        S
    ## 138     Died      1   male 37.00     1     0  53.1000        S
    ## 139     Died      3   male 16.00     0     0   9.2167        S
    ## 140     Died      1   male 24.00     0     0  79.2000        C
    ## 142 Survived      3 female 22.00     0     0   7.7500        S
    ## 143 Survived      3 female 24.00     1     0  15.8500        S
    ## 144     Died      3   male 19.00     0     0   6.7500        Q
    ## 145     Died      2   male 18.00     0     0  11.5000        S
    ## 146     Died      2   male 19.00     1     1  36.7500        S
    ## 147 Survived      3   male 27.00     0     0   7.7958        S
    ## 148     Died      3 female  9.00     2     2  34.3750        S
    ## 149     Died      2   male 36.50     0     2  26.0000        S
    ## 150     Died      2   male 42.00     0     0  13.0000        S
    ## 151     Died      2   male 51.00     0     0  12.5250        S
    ## 152 Survived      1 female 22.00     1     0  66.6000        S
    ## 153     Died      3   male 55.50     0     0   8.0500        S
    ## 154     Died      3   male 40.50     0     2  14.5000        S
    ## 156     Died      1   male 51.00     0     1  61.3792        C
    ## 157 Survived      3 female 16.00     0     0   7.7333        Q
    ## 158     Died      3   male 30.00     0     0   8.0500        S
    ## 161     Died      3   male 44.00     0     1  16.1000        S
    ## 162 Survived      2 female 40.00     0     0  15.7500        S
    ## 163     Died      3   male 26.00     0     0   7.7750        S
    ## 164     Died      3   male 17.00     0     0   8.6625        S
    ## 165     Died      3   male  1.00     4     1  39.6875        S
    ## 166 Survived      3   male  9.00     0     2  20.5250        S
    ## 168     Died      3 female 45.00     1     4  27.9000        S
    ## 170     Died      3   male 28.00     0     0  56.4958        S
    ## 171     Died      1   male 61.00     0     0  33.5000        S
    ## 172     Died      3   male  4.00     4     1  29.1250        Q
    ## 173 Survived      3 female  1.00     1     1  11.1333        S
    ## 174     Died      3   male 21.00     0     0   7.9250        S
    ## 175     Died      1   male 56.00     0     0  30.6958        C
    ## 176     Died      3   male 18.00     1     1   7.8542        S
    ## 178     Died      1 female 50.00     0     0  28.7125        C
    ## 179     Died      2   male 30.00     0     0  13.0000        S
    ## 180     Died      3   male 36.00     0     0   0.0000        S
    ## 183     Died      3   male  9.00     4     2  31.3875        S
    ## 184 Survived      2   male  1.00     2     1  39.0000        S
    ## 185 Survived      3 female  4.00     0     2  22.0250        S
    ## 188 Survived      1   male 45.00     0     0  26.5500        S
    ## 189     Died      3   male 40.00     1     1  15.5000        Q
    ## 190     Died      3   male 36.00     0     0   7.8958        S
    ## 191 Survived      2 female 32.00     0     0  13.0000        S
    ## 192     Died      2   male 19.00     0     0  13.0000        S
    ## 193 Survived      3 female 19.00     1     0   7.8542        S
    ## 194 Survived      2   male  3.00     1     1  26.0000        S
    ## 195 Survived      1 female 44.00     0     0  27.7208        C
    ## 196 Survived      1 female 58.00     0     0 146.5208        C
    ## 198     Died      3   male 42.00     0     1   8.4042        S
    ## 200     Died      2 female 24.00     0     0  13.0000        S
    ## 201     Died      3   male 28.00     0     0   9.5000        S
    ## 203     Died      3   male 34.00     0     0   6.4958        S
    ## 204     Died      3   male 45.50     0     0   7.2250        C
    ## 205 Survived      3   male 18.00     0     0   8.0500        S
    ## 206     Died      3 female  2.00     0     1  10.4625        S
    ## 207     Died      3   male 32.00     1     0  15.8500        S
    ## 208 Survived      3   male 26.00     0     0  18.7875        C
    ## 209 Survived      3 female 16.00     0     0   7.7500        Q
    ## 210 Survived      1   male 40.00     0     0  31.0000        C
    ## 211     Died      3   male 24.00     0     0   7.0500        S
    ## 212 Survived      2 female 35.00     0     0  21.0000        S
    ## 213     Died      3   male 22.00     0     0   7.2500        S
    ## 214     Died      2   male 30.00     0     0  13.0000        S
    ## 216 Survived      1 female 31.00     1     0 113.2750        C
    ## 217 Survived      3 female 27.00     0     0   7.9250        S
    ## 218     Died      2   male 42.00     1     0  27.0000        S
    ## 219 Survived      1 female 32.00     0     0  76.2917        C
    ## 220     Died      2   male 30.00     0     0  10.5000        S
    ## 221 Survived      3   male 16.00     0     0   8.0500        S
    ## 222     Died      2   male 27.00     0     0  13.0000        S
    ## 223     Died      3   male 51.00     0     0   8.0500        S
    ## 225 Survived      1   male 38.00     1     0  90.0000        S
    ## 226     Died      3   male 22.00     0     0   9.3500        S
    ## 227 Survived      2   male 19.00     0     0  10.5000        S
    ## 228     Died      3   male 20.50     0     0   7.2500        S
    ## 229     Died      2   male 18.00     0     0  13.0000        S
    ## 231 Survived      1 female 35.00     1     0  83.4750        S
    ## 232     Died      3   male 29.00     0     0   7.7750        S
    ## 233     Died      2   male 59.00     0     0  13.5000        S
    ## 234 Survived      3 female  5.00     4     2  31.3875        S
    ## 235     Died      2   male 24.00     0     0  10.5000        S
    ## 237     Died      2   male 44.00     1     0  26.0000        S
    ## 238 Survived      2 female  8.00     0     2  26.2500        S
    ## 239     Died      2   male 19.00     0     0  10.5000        S
    ## 240     Died      2   male 33.00     0     0  12.2750        S
    ## 243     Died      2   male 29.00     0     0  10.5000        S
    ## 244     Died      3   male 22.00     0     0   7.1250        S
    ## 245     Died      3   male 30.00     0     0   7.2250        C
    ## 246     Died      1   male 44.00     2     0  90.0000        Q
    ## 247     Died      3 female 25.00     0     0   7.7750        S
    ## 248 Survived      2 female 24.00     0     2  14.5000        S
    ## 249 Survived      1   male 37.00     1     1  52.5542        S
    ## 250     Died      2   male 54.00     1     0  26.0000        S
    ## 252     Died      3 female 29.00     1     1  10.4625        S
    ## 253     Died      1   male 62.00     0     0  26.5500        S
    ## 254     Died      3   male 30.00     1     0  16.1000        S
    ## 255     Died      3 female 41.00     0     2  20.2125        S
    ## 256 Survived      3 female 29.00     0     2  15.2458        C
    ## 258 Survived      1 female 30.00     0     0  86.5000        S
    ## 259 Survived      1 female 35.00     0     0 512.3292        C
    ## 260 Survived      2 female 50.00     0     1  26.0000        S
    ## 262 Survived      3   male  3.00     4     2  31.3875        S
    ## 263     Died      1   male 52.00     1     1  79.6500        S
    ## 264     Died      1   male 40.00     0     0   0.0000        S
    ## 266     Died      2   male 36.00     0     0  10.5000        S
    ## 267     Died      3   male 16.00     4     1  39.6875        S
    ## 268 Survived      3   male 25.00     1     0   7.7750        S
    ## 269 Survived      1 female 58.00     0     1 153.4625        S
    ## 270 Survived      1 female 35.00     0     0 135.6333        S
    ## 272 Survived      3   male 25.00     0     0   0.0000        S
    ## 273 Survived      2 female 41.00     0     1  19.5000        S
    ## 274     Died      1   male 37.00     0     1  29.7000        C
    ## 276 Survived      1 female 63.00     1     0  77.9583        S
    ## 277     Died      3 female 45.00     0     0   7.7500        S
    ## 279     Died      3   male  7.00     4     1  29.1250        Q
    ## 280 Survived      3 female 35.00     1     1  20.2500        S
    ## 281     Died      3   male 65.00     0     0   7.7500        Q
    ## 282     Died      3   male 28.00     0     0   7.8542        S
    ## 283     Died      3   male 16.00     0     0   9.5000        S
    ## 284 Survived      3   male 19.00     0     0   8.0500        S
    ## 286     Died      3   male 33.00     0     0   8.6625        C
    ## 287 Survived      3   male 30.00     0     0   9.5000        S
    ## 288     Died      3   male 22.00     0     0   7.8958        S
    ## 289 Survived      2   male 42.00     0     0  13.0000        S
    ## 290 Survived      3 female 22.00     0     0   7.7500        Q
    ## 291 Survived      1 female 26.00     0     0  78.8500        S
    ## 292 Survived      1 female 19.00     1     0  91.0792        C
    ## 293     Died      2   male 36.00     0     0  12.8750        C
    ## 294     Died      3 female 24.00     0     0   8.8500        S
    ## 295     Died      3   male 24.00     0     0   7.8958        S
    ## 297     Died      3   male 23.50     0     0   7.2292        C
    ## 298     Died      1 female  2.00     1     2 151.5500        S
    ## 300 Survived      1 female 50.00     0     1 247.5208        C
    ## 303     Died      3   male 19.00     0     0   0.0000        S
    ## 306 Survived      1   male  0.92     1     2 151.5500        S
    ## 308 Survived      1 female 17.00     1     0 108.9000        C
    ## 309     Died      2   male 30.00     1     0  24.0000        C
    ## 310 Survived      1 female 30.00     0     0  56.9292        C
    ## 311 Survived      1 female 24.00     0     0  83.1583        C
    ## 312 Survived      1 female 18.00     2     2 262.3750        C
    ## 313     Died      2 female 26.00     1     1  26.0000        S
    ## 314     Died      3   male 28.00     0     0   7.8958        S
    ## 315     Died      2   male 43.00     1     1  26.2500        S
    ## 316 Survived      3 female 26.00     0     0   7.8542        S
    ## 317 Survived      2 female 24.00     1     0  26.0000        S
    ## 318     Died      2   male 54.00     0     0  14.0000        S
    ## 319 Survived      1 female 31.00     0     2 164.8667        S
    ## 320 Survived      1 female 40.00     1     1 134.5000        C
    ## 321     Died      3   male 22.00     0     0   7.2500        S
    ## 322     Died      3   male 27.00     0     0   7.8958        S
    ## 323 Survived      2 female 30.00     0     0  12.3500        Q
    ## 324 Survived      2 female 22.00     1     1  29.0000        S
    ## 326 Survived      1 female 36.00     0     0 135.6333        C
    ## 327     Died      3   male 61.00     0     0   6.2375        S
    ## 328 Survived      2 female 36.00     0     0  13.0000        S
    ## 329 Survived      3 female 31.00     1     1  20.5250        S
    ## 330 Survived      1 female 16.00     0     1  57.9792        C
    ## 332     Died      1   male 45.50     0     0  28.5000        S
    ## 333     Died      1   male 38.00     0     1 153.4625        S
    ## 334     Died      3   male 16.00     2     0  18.0000        S
    ## 337     Died      1   male 29.00     1     0  66.6000        S
    ## 338 Survived      1 female 41.00     0     0 134.5000        C
    ## 339 Survived      3   male 45.00     0     0   8.0500        S
    ## 340     Died      1   male 45.00     0     0  35.5000        S
    ## 341 Survived      2   male  2.00     1     1  26.0000        S
    ## 342 Survived      1 female 24.00     3     2 263.0000        S
    ## 343     Died      2   male 28.00     0     0  13.0000        S
    ## 344     Died      2   male 25.00     0     0  13.0000        S
    ## 345     Died      2   male 36.00     0     0  13.0000        S
    ## 346 Survived      2 female 24.00     0     0  13.0000        S
    ## 347 Survived      2 female 40.00     0     0  13.0000        S
    ## 349 Survived      3   male  3.00     1     1  15.9000        S
    ## 350     Died      3   male 42.00     0     0   8.6625        S
    ## 351     Died      3   male 23.00     0     0   9.2250        S
    ## 353     Died      3   male 15.00     1     1   7.2292        C
    ## 354     Died      3   male 25.00     1     0  17.8000        S
    ## 356     Died      3   male 28.00     0     0   9.5000        S
    ## 357 Survived      1 female 22.00     0     1  55.0000        S
    ## 358     Died      2 female 38.00     0     0  13.0000        S
    ## 361     Died      3   male 40.00     1     4  27.9000        S
    ## 362     Died      2   male 29.00     1     0  27.7208        C
    ## 363     Died      3 female 45.00     0     1  14.4542        C
    ## 364     Died      3   male 35.00     0     0   7.0500        S
    ## 366     Died      3   male 30.00     0     0   7.2500        S
    ## 367 Survived      1 female 60.00     1     0  75.2500        C
    ## 370 Survived      1 female 24.00     0     0  69.3000        C
    ## 371 Survived      1   male 25.00     1     0  55.4417        C
    ## 372     Died      3   male 18.00     1     0   6.4958        S
    ## 373     Died      3   male 19.00     0     0   8.0500        S
    ## 374     Died      1   male 22.00     0     0 135.6333        C
    ## 375     Died      3 female  3.00     3     1  21.0750        S
    ## 377 Survived      3 female 22.00     0     0   7.2500        S
    ## 378     Died      1   male 27.00     0     2 211.5000        C
    ## 379     Died      3   male 20.00     0     0   4.0125        C
    ## 380     Died      3   male 19.00     0     0   7.7750        S
    ## 381 Survived      1 female 42.00     0     0 227.5250        C
    ## 382 Survived      3 female  1.00     0     2  15.7417        C
    ## 383     Died      3   male 32.00     0     0   7.9250        S
    ## 384 Survived      1 female 35.00     1     0  52.0000        S
    ## 386     Died      2   male 18.00     0     0  73.5000        S
    ## 387     Died      3   male  1.00     5     2  46.9000        S
    ## 388 Survived      2 female 36.00     0     0  13.0000        S
    ## 390 Survived      2 female 17.00     0     0  12.0000        C
    ## 391 Survived      1   male 36.00     1     2 120.0000        S
    ## 392 Survived      3   male 21.00     0     0   7.7958        S
    ## 393     Died      3   male 28.00     2     0   7.9250        S
    ## 394 Survived      1 female 23.00     1     0 113.2750        C
    ## 395 Survived      3 female 24.00     0     2  16.7000        S
    ## 396     Died      3   male 22.00     0     0   7.7958        S
    ## 397     Died      3 female 31.00     0     0   7.8542        S
    ## 398     Died      2   male 46.00     0     0  26.0000        S
    ## 399     Died      2   male 23.00     0     0  10.5000        S
    ## 400 Survived      2 female 28.00     0     0  12.6500        S
    ## 401 Survived      3   male 39.00     0     0   7.9250        S
    ## 402     Died      3   male 26.00     0     0   8.0500        S
    ## 403     Died      3 female 21.00     1     0   9.8250        S
    ## 404     Died      3   male 28.00     1     0  15.8500        S
    ## 405     Died      3 female 20.00     0     0   8.6625        S
    ## 406     Died      2   male 34.00     1     0  21.0000        S
    ## 407     Died      3   male 51.00     0     0   7.7500        S
    ## 408 Survived      2   male  3.00     1     1  18.7500        S
    ## 409     Died      3   male 21.00     0     0   7.7750        S
    ## 413 Survived      1 female 33.00     1     0  90.0000        Q
    ## 415 Survived      3   male 44.00     0     0   7.9250        S
    ## 417 Survived      2 female 34.00     1     1  32.5000        S
    ## 418 Survived      2 female 18.00     0     2  13.0000        S
    ## 419     Died      2   male 30.00     0     0  13.0000        S
    ## 420     Died      3 female 10.00     0     2  24.1500        S
    ## 422     Died      3   male 21.00     0     0   7.7333        Q
    ## 423     Died      3   male 29.00     0     0   7.8750        S
    ## 424     Died      3 female 28.00     1     1  14.4000        S
    ## 425     Died      3   male 18.00     1     1  20.2125        S
    ## 427 Survived      2 female 28.00     1     0  26.0000        S
    ## 428 Survived      2 female 19.00     0     0  26.0000        S
    ## 430 Survived      3   male 32.00     0     0   8.0500        S
    ## 431 Survived      1   male 28.00     0     0  26.5500        S
    ## 433 Survived      2 female 42.00     1     0  26.0000        S
    ## 434     Died      3   male 17.00     0     0   7.1250        S
    ## 435     Died      1   male 50.00     1     0  55.9000        S
    ## 436 Survived      1 female 14.00     1     2 120.0000        S
    ## 437     Died      3 female 21.00     2     2  34.3750        S
    ## 438 Survived      2 female 24.00     2     3  18.7500        S
    ## 439     Died      1   male 64.00     1     4 263.0000        S
    ## 440     Died      2   male 31.00     0     0  10.5000        S
    ## 441 Survived      2 female 45.00     1     1  26.2500        S
    ## 442     Died      3   male 20.00     0     0   9.5000        S
    ## 443     Died      3   male 25.00     1     0   7.7750        S
    ## 444 Survived      2 female 28.00     0     0  13.0000        S
    ## 446 Survived      1   male  4.00     0     2  81.8583        S
    ## 447 Survived      2 female 13.00     0     1  19.5000        S
    ## 448 Survived      1   male 34.00     0     0  26.5500        S
    ## 449 Survived      3 female  5.00     2     1  19.2583        C
    ## 450 Survived      1   male 52.00     0     0  30.5000        S
    ## 451     Died      2   male 36.00     1     2  27.7500        S
    ## 453     Died      1   male 30.00     0     0  27.7500        C
    ## 454 Survived      1   male 49.00     1     0  89.1042        C
    ## 456 Survived      3   male 29.00     0     0   7.8958        C
    ## 457     Died      1   male 65.00     0     0  26.5500        S
    ## 459 Survived      2 female 50.00     0     0  10.5000        S
    ## 461 Survived      1   male 48.00     0     0  26.5500        S
    ## 462     Died      3   male 34.00     0     0   8.0500        S
    ## 463     Died      1   male 47.00     0     0  38.5000        S
    ## 464     Died      2   male 48.00     0     0  13.0000        S
    ## 466     Died      3   male 38.00     0     0   7.0500        S
    ## 468     Died      1   male 56.00     0     0  26.5500        S
    ## 470 Survived      3 female  0.75     2     1  19.2583        C
    ## 472     Died      3   male 38.00     0     0   8.6625        S
    ## 473 Survived      2 female 33.00     1     2  27.7500        S
    ## 474 Survived      2 female 23.00     0     0  13.7917        C
    ## 475     Died      3 female 22.00     0     0   9.8375        S
    ## 477     Died      2   male 34.00     1     0  21.0000        S
    ## 478     Died      3   male 29.00     1     0   7.0458        S
    ## 479     Died      3   male 22.00     0     0   7.5208        S
    ## 480 Survived      3 female  2.00     0     1  12.2875        S
    ## 481     Died      3   male  9.00     5     2  46.9000        S
    ## 483     Died      3   male 50.00     0     0   8.0500        S
    ## 484 Survived      3 female 63.00     0     0   9.5875        S
    ## 485 Survived      1   male 25.00     1     0  91.0792        C
    ## 487 Survived      1 female 35.00     1     0  90.0000        S
    ## 488     Died      1   male 58.00     0     0  29.7000        C
    ## 489     Died      3   male 30.00     0     0   8.0500        S
    ## 490 Survived      3   male  9.00     1     1  15.9000        S
    ## 492     Died      3   male 21.00     0     0   7.2500        S
    ## 493     Died      1   male 55.00     0     0  30.5000        S
    ## 494     Died      1   male 71.00     0     0  49.5042        C
    ## 495     Died      3   male 21.00     0     0   8.0500        S
    ## 497 Survived      1 female 54.00     1     0  78.2667        C
    ## 499     Died      1 female 25.00     1     2 151.5500        S
    ## 500     Died      3   male 24.00     0     0   7.7958        S
    ## 501     Died      3   male 17.00     0     0   8.6625        S
    ## 502     Died      3 female 21.00     0     0   7.7500        Q
    ## 504     Died      3 female 37.00     0     0   9.5875        S
    ## 505 Survived      1 female 16.00     0     0  86.5000        S
    ## 506     Died      1   male 18.00     1     0 108.9000        C
    ## 507 Survived      2 female 33.00     0     2  26.0000        S
    ## 509     Died      3   male 28.00     0     0  22.5250        S
    ## 510 Survived      3   male 26.00     0     0  56.4958        S
    ## 511 Survived      3   male 29.00     0     0   7.7500        Q
    ## 513 Survived      1   male 36.00     0     0  26.2875        S
    ## 514 Survived      1 female 54.00     1     0  59.4000        C
    ## 515     Died      3   male 24.00     0     0   7.4958        S
    ## 516     Died      1   male 47.00     0     0  34.0208        S
    ## 517 Survived      2 female 34.00     0     0  10.5000        S
    ## 519 Survived      2 female 36.00     1     0  26.0000        S
    ## 520     Died      3   male 32.00     0     0   7.8958        S
    ## 521 Survived      1 female 30.00     0     0  93.5000        S
    ## 522     Died      3   male 22.00     0     0   7.8958        S
    ## 524 Survived      1 female 44.00     0     1  57.9792        C
    ## 526     Died      3   male 40.50     0     0   7.7500        Q
    ## 527 Survived      2 female 50.00     0     0  10.5000        S
    ## 529     Died      3   male 39.00     0     0   7.9250        S
    ## 530     Died      2   male 23.00     2     1  11.5000        S
    ## 531 Survived      2 female  2.00     1     1  26.0000        S
    ## 533     Died      3   male 17.00     1     1   7.2292        C
    ## 535     Died      3 female 30.00     0     0   8.6625        S
    ## 536 Survived      2 female  7.00     0     2  26.2500        S
    ## 537     Died      1   male 45.00     0     0  26.5500        S
    ## 538 Survived      1 female 30.00     0     0 106.4250        C
    ## 540 Survived      1 female 22.00     0     2  49.5000        C
    ## 541 Survived      1 female 36.00     0     2  71.0000        S
    ## 542     Died      3 female  9.00     4     2  31.2750        S
    ## 543     Died      3 female 11.00     4     2  31.2750        S
    ## 544 Survived      2   male 32.00     1     0  26.0000        S
    ## 545     Died      1   male 50.00     1     0 106.4250        C
    ## 546     Died      1   male 64.00     0     0  26.0000        S
    ## 547 Survived      2 female 19.00     1     0  26.0000        S
    ## 549     Died      3   male 33.00     1     1  20.5250        S
    ## 550 Survived      2   male  8.00     1     1  36.7500        S
    ## 551 Survived      1   male 17.00     0     2 110.8833        C
    ## 552     Died      2   male 27.00     0     0  26.0000        S
    ## 554 Survived      3   male 22.00     0     0   7.2250        C
    ## 555 Survived      3 female 22.00     0     0   7.7750        S
    ## 556     Died      1   male 62.00     0     0  26.5500        S
    ## 557 Survived      1 female 48.00     1     0  39.6000        C
    ## 559 Survived      1 female 39.00     1     1  79.6500        S
    ## 560 Survived      3 female 36.00     1     0  17.4000        S
    ## 562     Died      3   male 40.00     0     0   7.8958        S
    ## 563     Died      2   male 28.00     0     0  13.5000        S
    ## 566     Died      3   male 24.00     2     0  24.1500        S
    ## 567     Died      3   male 19.00     0     0   7.8958        S
    ## 568     Died      3 female 29.00     0     4  21.0750        S
    ## 570 Survived      3   male 32.00     0     0   7.8542        S
    ## 571 Survived      2   male 62.00     0     0  10.5000        S
    ## 572 Survived      1 female 53.00     2     0  51.4792        S
    ## 573 Survived      1   male 36.00     0     0  26.3875        S
    ## 575     Died      3   male 16.00     0     0   8.0500        S
    ## 576     Died      3   male 19.00     0     0  14.5000        S
    ## 577 Survived      2 female 34.00     0     0  13.0000        S
    ## 578 Survived      1 female 39.00     1     0  55.9000        S
    ## 580 Survived      3   male 32.00     0     0   7.9250        S
    ## 581 Survived      2 female 25.00     1     1  30.0000        S
    ## 582 Survived      1 female 39.00     1     1 110.8833        C
    ## 583     Died      2   male 54.00     0     0  26.0000        S
    ## 584     Died      1   male 36.00     0     0  40.1250        C
    ## 586 Survived      1 female 18.00     0     2  79.6500        S
    ## 587     Died      2   male 47.00     0     0  15.0000        S
    ## 588 Survived      1   male 60.00     1     1  79.2000        C
    ## 589     Died      3   male 22.00     0     0   8.0500        S
    ## 591     Died      3   male 35.00     0     0   7.1250        S
    ## 592 Survived      1 female 52.00     1     0  78.2667        C
    ## 593     Died      3   male 47.00     0     0   7.2500        S
    ## 595     Died      2   male 37.00     1     0  26.0000        S
    ## 596     Died      3   male 36.00     1     1  24.1500        S
    ## 598     Died      3   male 49.00     0     0   0.0000        S
    ## 600 Survived      1   male 49.00     1     0  56.9292        C
    ## 601 Survived      2 female 24.00     2     1  27.0000        S
    ## 604     Died      3   male 44.00     0     0   8.0500        S
    ## 605 Survived      1   male 35.00     0     0  26.5500        C
    ## 606     Died      3   male 36.00     1     0  15.5500        S
    ## 607     Died      3   male 30.00     0     0   7.8958        S
    ## 608 Survived      1   male 27.00     0     0  30.5000        S
    ## 609 Survived      2 female 22.00     1     2  41.5792        C
    ## 610 Survived      1 female 40.00     0     0 153.4625        S
    ## 611     Died      3 female 39.00     1     5  31.2750        S
    ## 615     Died      3   male 35.00     0     0   8.0500        S
    ## 616 Survived      2 female 24.00     1     2  65.0000        S
    ## 617     Died      3   male 34.00     1     1  14.4000        S
    ## 618     Died      3 female 26.00     1     0  16.1000        S
    ## 619 Survived      2 female  4.00     2     1  39.0000        S
    ## 620     Died      2   male 26.00     0     0  10.5000        S
    ## 621     Died      3   male 27.00     1     0  14.4542        C
    ## 622 Survived      1   male 42.00     1     0  52.5542        S
    ## 623 Survived      3   male 20.00     1     1  15.7417        C
    ## 624     Died      3   male 21.00     0     0   7.8542        S
    ## 625     Died      3   male 21.00     0     0  16.1000        S
    ## 626     Died      1   male 61.00     0     0  32.3208        S
    ## 627     Died      2   male 57.00     0     0  12.3500        Q
    ## 628 Survived      1 female 21.00     0     0  77.9583        S
    ## 629     Died      3   male 26.00     0     0   7.8958        S
    ## 631 Survived      1   male 80.00     0     0  30.0000        S
    ## 632     Died      3   male 51.00     0     0   7.0542        S
    ## 633 Survived      1   male 32.00     0     0  30.5000        C
    ## 635     Died      3 female  9.00     3     2  27.9000        S
    ## 636 Survived      2 female 28.00     0     0  13.0000        S
    ## 637     Died      3   male 32.00     0     0   7.9250        S
    ## 638     Died      2   male 31.00     1     1  26.2500        S
    ## 639     Died      3 female 41.00     0     5  39.6875        S
    ## 641     Died      3   male 20.00     0     0   7.8542        S
    ## 642 Survived      1 female 24.00     0     0  69.3000        C
    ## 643     Died      3 female  2.00     3     2  27.9000        S
    ## 645 Survived      3 female  0.75     2     1  19.2583        C
    ## 646 Survived      1   male 48.00     1     0  76.7292        C
    ## 647     Died      3   male 19.00     0     0   7.8958        S
    ## 648 Survived      1   male 56.00     0     0  35.5000        C
    ## 650 Survived      3 female 23.00     0     0   7.5500        S
    ## 652 Survived      2 female 18.00     0     1  23.0000        S
    ## 653     Died      3   male 21.00     0     0   8.4333        S
    ## 655     Died      3 female 18.00     0     0   6.7500        Q
    ## 656     Died      2   male 24.00     2     0  73.5000        S
    ## 658     Died      3 female 32.00     1     1  15.5000        Q
    ## 659     Died      2   male 23.00     0     0  13.0000        S
    ## 660     Died      1   male 58.00     0     2 113.2750        C
    ## 661 Survived      1   male 50.00     2     0 133.6500        S
    ## 662     Died      3   male 40.00     0     0   7.2250        C
    ## 663     Died      1   male 47.00     0     0  25.5875        S
    ## 664     Died      3   male 36.00     0     0   7.4958        S
    ## 665 Survived      3   male 20.00     1     0   7.9250        S
    ## 666     Died      2   male 32.00     2     0  73.5000        S
    ## 667     Died      2   male 25.00     0     0  13.0000        S
    ## 669     Died      3   male 43.00     0     0   8.0500        S
    ## 671 Survived      2 female 40.00     1     1  39.0000        S
    ## 672     Died      1   male 31.00     1     0  52.0000        S
    ## 673     Died      2   male 70.00     0     0  10.5000        S
    ## 674 Survived      2   male 31.00     0     0  13.0000        S
    ## 676     Died      3   male 18.00     0     0   7.7750        S
    ## 677     Died      3   male 24.50     0     0   8.0500        S
    ## 678 Survived      3 female 18.00     0     0   9.8417        S
    ## 679     Died      3 female 43.00     1     6  46.9000        S
    ## 680 Survived      1   male 36.00     0     1 512.3292        C
    ## 682 Survived      1   male 27.00     0     0  76.7292        C
    ## 683     Died      3   male 20.00     0     0   9.2250        S
    ## 684     Died      3   male 14.00     5     2  46.9000        S
    ## 685     Died      2   male 60.00     1     1  39.0000        S
    ## 686     Died      2   male 25.00     1     2  41.5792        C
    ## 687     Died      3   male 14.00     4     1  39.6875        S
    ## 688     Died      3   male 19.00     0     0  10.1708        S
    ## 689     Died      3   male 18.00     0     0   7.7958        S
    ## 690 Survived      1 female 15.00     0     1 211.3375        S
    ## 691 Survived      1   male 31.00     1     0  57.0000        S
    ## 692 Survived      3 female  4.00     0     1  13.4167        C
    ## 694     Died      3   male 25.00     0     0   7.2250        C
    ## 695     Died      1   male 60.00     0     0  26.5500        S
    ## 696     Died      2   male 52.00     0     0  13.5000        S
    ## 697     Died      3   male 44.00     0     0   8.0500        S
    ## 699     Died      1   male 49.00     1     1 110.8833        C
    ## 700     Died      3   male 42.00     0     0   7.6500        S
    ## 701 Survived      1 female 18.00     1     0 227.5250        C
    ## 702 Survived      1   male 35.00     0     0  26.2875        S
    ## 703     Died      3 female 18.00     0     1  14.4542        C
    ## 704     Died      3   male 25.00     0     0   7.7417        Q
    ## 705     Died      3   male 26.00     1     0   7.8542        S
    ## 706     Died      2   male 39.00     0     0  26.0000        S
    ## 707 Survived      2 female 45.00     0     0  13.5000        S
    ## 708 Survived      1   male 42.00     0     0  26.2875        S
    ## 709 Survived      1 female 22.00     0     0 151.5500        S
    ## 711 Survived      1 female 24.00     0     0  49.5042        C
    ## 713 Survived      1   male 48.00     1     0  52.0000        S
    ## 714     Died      3   male 29.00     0     0   9.4833        S
    ## 715     Died      2   male 52.00     0     0  13.0000        S
    ## 716     Died      3   male 19.00     0     0   7.6500        S
    ## 717 Survived      1 female 38.00     0     0 227.5250        C
    ## 718 Survived      2 female 27.00     0     0  10.5000        S
    ## 720     Died      3   male 33.00     0     0   7.7750        S
    ## 721 Survived      2 female  6.00     0     1  33.0000        S
    ## 722     Died      3   male 17.00     1     0   7.0542        S
    ## 723     Died      2   male 34.00     0     0  13.0000        S
    ## 724     Died      2   male 50.00     0     0  13.0000        S
    ## 725 Survived      1   male 27.00     1     0  53.1000        S
    ## 726     Died      3   male 20.00     0     0   8.6625        S
    ## 727 Survived      2 female 30.00     3     0  21.0000        S
    ## 729     Died      2   male 25.00     1     0  26.0000        S
    ## 730     Died      3 female 25.00     1     0   7.9250        S
    ## 731 Survived      1 female 29.00     0     0 211.3375        S
    ## 732     Died      3   male 11.00     0     0  18.7875        C
    ## 734     Died      2   male 23.00     0     0  13.0000        S
    ## 735     Died      2   male 23.00     0     0  13.0000        S
    ## 736     Died      3   male 28.50     0     0  16.1000        S
    ## 737     Died      3 female 48.00     1     3  34.3750        S
    ## 738 Survived      1   male 35.00     0     0 512.3292        C
    ## 742     Died      1   male 36.00     1     0  78.8500        S
    ## 743 Survived      1 female 21.00     2     2 262.3750        C
    ## 744     Died      3   male 24.00     1     0  16.1000        S
    ## 745 Survived      3   male 31.00     0     0   7.9250        S
    ## 746     Died      1   male 70.00     1     1  71.0000        S
    ## 747     Died      3   male 16.00     1     1  20.2500        S
    ## 748 Survived      2 female 30.00     0     0  13.0000        S
    ## 749     Died      1   male 19.00     1     0  53.1000        S
    ## 750     Died      3   male 31.00     0     0   7.7500        Q
    ## 751 Survived      2 female  4.00     1     1  23.0000        S
    ## 752 Survived      3   male  6.00     0     1  12.4750        S
    ## 753     Died      3   male 33.00     0     0   9.5000        S
    ## 754     Died      3   male 23.00     0     0   7.8958        S
    ## 755 Survived      2 female 48.00     1     2  65.0000        S
    ## 756 Survived      2   male  0.67     1     1  14.5000        S
    ## 757     Died      3   male 28.00     0     0   7.7958        S
    ## 758     Died      2   male 18.00     0     0  11.5000        S
    ## 759     Died      3   male 34.00     0     0   8.0500        S
    ## 760 Survived      1 female 33.00     0     0  86.5000        S
    ## 762     Died      3   male 41.00     0     0   7.1250        S
    ## 763 Survived      3   male 20.00     0     0   7.2292        C
    ## 764 Survived      1 female 36.00     1     2 120.0000        S
    ## 765     Died      3   male 16.00     0     0   7.7750        S
    ## 766 Survived      1 female 51.00     1     0  77.9583        S
    ## 768     Died      3 female 30.50     0     0   7.7500        Q
    ## 770     Died      3   male 32.00     0     0   8.3625        S
    ## 771     Died      3   male 24.00     0     0   9.5000        S
    ## 772     Died      3   male 48.00     0     0   7.8542        S
    ## 773     Died      2 female 57.00     0     0  10.5000        S
    ## 775 Survived      2 female 54.00     1     3  23.0000        S
    ## 776     Died      3   male 18.00     0     0   7.7500        S
    ## 778 Survived      3 female  5.00     0     0  12.4750        S
    ## 780 Survived      1 female 43.00     0     1 211.3375        S
    ## 781 Survived      3 female 13.00     0     0   7.2292        C
    ## 782 Survived      1 female 17.00     1     0  57.0000        S
    ## 783     Died      1   male 29.00     0     0  30.0000        S
    ## 785     Died      3   male 25.00     0     0   7.0500        S
    ## 786     Died      3   male 25.00     0     0   7.2500        S
    ## 787 Survived      3 female 18.00     0     0   7.4958        S
    ## 788     Died      3   male  8.00     4     1  29.1250        Q
    ## 789 Survived      3   male  1.00     1     2  20.5750        S
    ## 790     Died      1   male 46.00     0     0  79.2000        C
    ## 792     Died      2   male 16.00     0     0  26.0000        S
    ## 795     Died      3   male 25.00     0     0   7.8958        S
    ## 796     Died      2   male 39.00     0     0  13.0000        S
    ## 797 Survived      1 female 49.00     0     0  25.9292        S
    ## 798 Survived      3 female 31.00     0     0   8.6833        S
    ## 799     Died      3   male 30.00     0     0   7.2292        C
    ## 800     Died      3 female 30.00     1     1  24.1500        S
    ## 801     Died      2   male 34.00     0     0  13.0000        S
    ## 802 Survived      2 female 31.00     1     1  26.2500        S
    ## 803 Survived      1   male 11.00     1     2 120.0000        S
    ## 804 Survived      3   male  0.42     0     1   8.5167        C
    ## 805 Survived      3   male 27.00     0     0   6.9750        S
    ## 806     Died      3   male 31.00     0     0   7.7750        S
    ## 807     Died      1   male 39.00     0     0   0.0000        S
    ## 808     Died      3 female 18.00     0     0   7.7750        S
    ## 809     Died      2   male 39.00     0     0  13.0000        S
    ## 810 Survived      1 female 33.00     1     0  53.1000        S
    ## 811     Died      3   male 26.00     0     0   7.8875        S
    ## 812     Died      3   male 39.00     0     0  24.1500        S
    ## 813     Died      2   male 35.00     0     0  10.5000        S
    ## 814     Died      3 female  6.00     4     2  31.2750        S
    ## 815     Died      3   male 30.50     0     0   8.0500        S
    ## 817     Died      3 female 23.00     0     0   7.9250        S
    ## 818     Died      2   male 31.00     1     1  37.0042        C
    ## 819     Died      3   male 43.00     0     0   6.4500        S
    ## 820     Died      3   male 10.00     3     2  27.9000        S
    ## 821 Survived      1 female 52.00     1     1  93.5000        S
    ## 822 Survived      3   male 27.00     0     0   8.6625        S
    ## 823     Died      1   male 38.00     0     0   0.0000        S
    ## 824 Survived      3 female 27.00     0     1  12.4750        S
    ## 825     Died      3   male  2.00     4     1  39.6875        S
    ## 828 Survived      2   male  1.00     0     2  37.0042        C
    ## 830 Survived      1 female 62.00     0     0  80.0000         
    ## 831 Survived      3 female 15.00     1     0  14.4542        C
    ## 832 Survived      2   male  0.83     1     1  18.7500        S
    ## 834     Died      3   male 23.00     0     0   7.8542        S
    ## 835     Died      3   male 18.00     0     0   8.3000        S
    ## 836 Survived      1 female 39.00     1     1  83.1583        C
    ## 837     Died      3   male 21.00     0     0   8.6625        S
    ## 839 Survived      3   male 32.00     0     0  56.4958        S
    ## 841     Died      3   male 20.00     0     0   7.9250        S
    ## 842     Died      2   male 16.00     0     0  10.5000        S
    ## 843 Survived      1 female 30.00     0     0  31.0000        C
    ## 844     Died      3   male 34.50     0     0   6.4375        C
    ## 845     Died      3   male 17.00     0     0   8.6625        S
    ## 846     Died      3   male 42.00     0     0   7.5500        S
    ## 848     Died      3   male 35.00     0     0   7.8958        C
    ## 849     Died      2   male 28.00     0     1  33.0000        S
    ## 851     Died      3   male  4.00     4     2  31.2750        S
    ## 852     Died      3   male 74.00     0     0   7.7750        S
    ## 853     Died      3 female  9.00     1     1  15.2458        C
    ## 854 Survived      1 female 16.00     0     1  39.4000        S
    ## 855     Died      2 female 44.00     1     0  26.0000        S
    ## 856 Survived      3 female 18.00     0     1   9.3500        S
    ## 857 Survived      1 female 45.00     1     1 164.8667        S
    ## 858 Survived      1   male 51.00     0     0  26.5500        S
    ## 859 Survived      3 female 24.00     0     3  19.2583        C
    ## 861     Died      3   male 41.00     2     0  14.1083        S
    ## 862     Died      2   male 21.00     1     0  11.5000        S
    ## 863 Survived      1 female 48.00     0     0  25.9292        S
    ## 865     Died      2   male 24.00     0     0  13.0000        S
    ## 866 Survived      2 female 42.00     0     0  13.0000        S
    ## 867 Survived      2 female 27.00     1     0  13.8583        C
    ## 868     Died      1   male 31.00     0     0  50.4958        S
    ## 870 Survived      3   male  4.00     1     1  11.1333        S
    ## 871     Died      3   male 26.00     0     0   7.8958        S
    ## 872 Survived      1 female 47.00     1     1  52.5542        S
    ## 873     Died      1   male 33.00     0     0   5.0000        S
    ## 874     Died      3   male 47.00     0     0   9.0000        S
    ## 875 Survived      2 female 28.00     1     0  24.0000        C
    ## 876 Survived      3 female 15.00     0     0   7.2250        C
    ## 877     Died      3   male 20.00     0     0   9.8458        S
    ## 878     Died      3   male 19.00     0     0   7.8958        S
    ## 880 Survived      1 female 56.00     0     1  83.1583        C
    ## 881 Survived      2 female 25.00     0     1  26.0000        S
    ## 882     Died      3   male 33.00     0     0   7.8958        S
    ## 883     Died      3 female 22.00     0     0  10.5167        S
    ## 884     Died      2   male 28.00     0     0  10.5000        S
    ## 885     Died      3   male 25.00     0     0   7.0500        S
    ## 886     Died      3 female 39.00     0     5  29.1250        Q
    ## 887     Died      2   male 27.00     0     0  13.0000        S
    ## 888 Survived      1 female 19.00     0     0  30.0000        S
    ## 890 Survived      1   male 26.00     0     0  30.0000        C
    ## 891     Died      3   male 32.00     0     0   7.7500        Q

``` r
#run rf for age + Pclass=====================================
age_class_rf <- train(Survived ~ Age + Pclass, data = titanic_rf_data,
                   method = "rf",
                   ntree = 500,
                   trControl = trainControl(method = "oob"))
```

    ## note: only 1 unique complexity parameters in default grid. Truncating the grid to 1 .

``` r
age_class_rf
```

    ## Random Forest 
    ## 
    ## 714 samples
    ##   2 predictor
    ##   2 classes: 'Died', 'Survived' 
    ## 
    ## No pre-processing
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.6862745  0.3284176
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 2

``` r
#run rf for sex + Pclass======================================
sex_class_rf <- train(Survived ~ Sex + Pclass, data = titanic_rf_data,
                   method = "rf",
                   ntree = 500,
                   trControl = trainControl(method = "oob"))
```

    ## note: only 1 unique complexity parameters in default grid. Truncating the grid to 1 .

``` r
sex_class_rf
```

    ## Random Forest 
    ## 
    ## 714 samples
    ##   2 predictor
    ##   2 classes: 'Died', 'Survived' 
    ## 
    ## No pre-processing
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.7913165  0.5341426
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 2

``` r
#run rf for sex + Pclass======================================
sex_class_age_rf <- train(Survived ~ Sex + Pclass + Age, data = titanic_rf_data,
                   method = "rf",
                   ntree = 500,
                   trControl = trainControl(method = "oob"))
```

    ## note: only 2 unique complexity parameters in default grid. Truncating the grid to 2 .

``` r
sex_class_age_rf
```

    ## Random Forest 
    ## 
    ## 714 samples
    ##   3 predictor
    ##   2 classes: 'Died', 'Survived' 
    ## 
    ## No pre-processing
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   2     0.8123249  0.5880509
    ##   3     0.8193277  0.6173315
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 3.

``` r
#Generate variable importance plots for each random forest model. Which variables seem the most important?
#age and class
randomForest::varImpPlot(age_class_rf$finalModel)
```

![](hw6_titanic_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
#sex and class
randomForest::varImpPlot(sex_class_rf$finalModel)
```

![](hw6_titanic_files/figure-markdown_github/unnamed-chunk-7-2.png)

``` r
#sex, class, and age
randomForest::varImpPlot(sex_class_age_rf$finalModel)
```

![](hw6_titanic_files/figure-markdown_github/unnamed-chunk-7-3.png)

``` r
#Calculate the out-of-bag error rate for each random forest model. Which performs the best?
#age and class
age_class_rf$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, ntree = 500, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 31.65%
    ## Confusion matrix:
    ##          Died Survived class.error
    ## Died      339       85   0.2004717
    ## Survived  141      149   0.4862069

``` r
print("OOB Estimate = 20.87%")
```

    ## [1] "OOB Estimate = 20.87%"

``` r
#sex and class
sex_class_rf$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, ntree = 500, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 20.87%
    ## Confusion matrix:
    ##          Died Survived class.error
    ## Died      415        9  0.02122642
    ## Survived  140      150  0.48275862

``` r
print("OOB Estimate = 20.87%")
```

    ## [1] "OOB Estimate = 20.87%"

``` r
#sex, class, and age
sex_class_age_rf$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, ntree = 500, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 18.91%
    ## Confusion matrix:
    ##          Died Survived class.error
    ## Died      375       49   0.1155660
    ## Survived   86      204   0.2965517

``` r
print("OOB Estimate = 18.35%")
```

    ## [1] "OOB Estimate = 18.35%"