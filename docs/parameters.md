# Parametre Rehberi

OptiProphet, Prophet yaklaşımından türetilen OptiWisdom OptiScorer içgörülerini
taşıyan parametrik bir altyapı sunar. Aşağıdaki tablolar en kritik ayarların
amacını, tipik aralıklarını ve ilgili bilimsel referansları listeler.

## Model Yapısı

| Parametre | Açıklama | Önerilen değerler | Referans |
| --- | --- | --- | --- |
| `n_changepoints` | Trend kırılmalarını yakalamak için kullanılan potansiyel nokta sayısı. | 5-25 arası, veri uzunluğuna göre. | Taylor & Letham (2018) |
| `seasonalities` | Her sezonsallık için periyot (`period`) ve Fourier derecesi (`order`). | Günlük/haftalık/aylık gibi bilinen periyotlar. | Taylor & Letham (2018) |
| `ar_order`, `ma_order` | Kısa vadeli bağımlılığı yakalayan AR/MA gecikmeleri. | AR: 1-5, MA: 0-3. | Box, Jenkins & Reinsel (2015) |
| `regressors` | Dışsal değişkenlerin listesi. | İş bağlamına uygun kolon adları. | Hyndman & Athanasopoulos (2021) |

## Tarihsel Bileşen Görünürlüğü

| Parametre | Açıklama |
| --- | --- |
| `historical_components` | `{"trend": True, "seasonality": False}` gibi anahtar/değer çiftleriyle `history_components()` çıktısındaki bileşenleri varsayılan olarak açıp kapatır.
| `history_components(include_components=..., component_overrides=...)` | Çağrı bazında toplu veya seçici görünürlük sağlar.
| `history_components(include_uncertainty=False)` | `yhat_lower`, `yhat_upper` ve kantil sütunlarını gizler.

## Gelecek Tahmin Kontrolleri

| Parametre | Açıklama |
| --- | --- |
| `forecast_components` | `predict()` çıktısındaki bileşen sütunlarının varsayılan olarak eklenip eklenmeyeceğini belirler.
| `predict(include_components=False)` | Gelecek tahminini yalın (`ds`, `yhat`) formatında döndürür.
| `predict(component_overrides={"seasonality": False})` | Yalnızca hedeflenen bileşenleri gizler.
| `predict(include_uncertainty=False)` | Güven aralıklarını ve kantil sütunlarını kaldırır.
| `predict(quantile_subset=[0.1, 0.5, 0.9])` | Yalnızca seçilen kantilleri döndürür; `self.quantiles` içinde olmalıdır.

## Backtest Stratejileri

| Strateji | Açıklama | Parametreler |
| --- | --- | --- |
| `expanding` | Her değerlendirme adımında eğitim penceresi genişler. | Varsayılan, klasik Prophet yaklaşımı. |
| `sliding` | Sabit uzunluklu bir pencere ilerletilir. | `window` parametresi pencere boyutunu belirler (varsayılan: `min_history`). |
| `anchored` | İlk `window` gözlemine sabitlenir ve her değerlendirme için aynı eğitim seti kullanılır. | Transfer öğrenimi benzeri sabit tabanlı senaryolar için. |

`backtest()` fonksiyonundaki `include_components`, `component_overrides`,
`include_uncertainty` ve `quantile_subset` parametreleri `predict()` ile aynı
anlamlara sahiptir ve değerlendirme dilimlerine ait tahmin çıktılarının
biçimlendirilmesini kontrol eder.

## Bilimsel Referanslar

* **Taylor, S. J., & Letham, B. (2018).** Forecasting at scale. *The American
  Statistician*, 72(1), 37-45.
* **Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).**
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
* **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and
  Practice* (3rd ed.). OTexts.

Bu referanslar OptiScorer çalışmalarında kullanılan bilimsel altyapıyı
belgeleyerek OptiProphet'in parametre tasarımlarının dayandığı kaynakları
paylaşır.
