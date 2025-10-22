# OptiProphet API Özeti

Bu belge OptiWisdom OptiScorer mirasını taşıyan OptiProphet kütüphanesinin
kamusal arayüzünü özetler. Aşağıdaki başlıklar `optitime` paketinin PyPI
dağıtımına hazır olacak şekilde planlandığını ve tüm işlevlerin saf Python ile
gerçekleştirildiğini hatırlatır.

## `optitime.OptiProphet`

### Yapılandırıcı (`__init__`)

```python
OptiProphet(
    *,
    n_changepoints: int = 15,
    seasonalities: Optional[Dict[str, Dict[str, float | int]]] = None,
    seasonality_mode: str = "additive",
    regressors: Optional[Iterable[str]] = None,
    ar_order: int = 2,
    ma_order: int = 1,
    interval_width: float = 0.8,
    quantiles: Iterable[float] = (0.1, 0.9),
    min_history: int = 30,
    min_success_r2: float = 0.1,
    max_mape: Optional[float] = 35.0,
    historical_components: Optional[Mapping[str, bool]] = None,
    forecast_components: bool = True,
    default_backtest_strategy: str = "expanding",
    default_backtest_window: Optional[int] = None,
)
```

* `historical_components`: eğitim geçmişindeki trend, sezonsallık, regresör,
  AR ve MA katkılarının varsayılan görünürlüğünü ayarlar.
* `forecast_components`: geleceğe yönelik tahminlerde bileşen sütunlarının
  varsayılan olarak eklenip eklenmeyeceğini belirler.
* `default_backtest_strategy` ve `default_backtest_window`: `backtest()`
  çağrısı sırasında varsayılan stratejiyi belirler (`expanding`, `sliding`,
  `anchored`).

### `fit(df: pd.DataFrame) -> OptiProphet`

`df` veri çerçevesinin `ds` (zaman damgası) ve `y` (hedef) sütunları içerdiğini
varsayar. Veri temizliği, interpolasyon ve yoğun hata kontrolü içerir.

### `make_future_dataframe(periods: int, freq: Optional[str] = None, include_history: bool = False)`

Prophet tarzı gelecek veri çerçevesi üretir. `include_history=True` olduğunda
eğitim zaman damgalarını da döndürür.

### `predict(...)`

```python
predict(
    future: Optional[pd.DataFrame] = None,
    *,
    include_history: bool = True,
    backcast: bool = False,
    include_components: Optional[bool] = None,
    component_overrides: Optional[Mapping[str, bool]] = None,
    include_uncertainty: bool = True,
    quantile_subset: Optional[Iterable[float]] = None,
)
```

* `include_components`: tüm bileşen sütunlarını topluca açıp kapatır.
* `component_overrides`: belirli bileşenleri (ör. `{"seasonality": False}`)
  devre dışı bırakır.
* `include_uncertainty`: güven aralıkları (`yhat_lower`, `yhat_upper`) ve
  seçili kantil sütunlarını (`yhat_q0.10` vb.) ekler veya kaldırır.
* `quantile_subset`: sadece belirtilen kantilleri döndürür.

### `history_components(...)`

```python
history_components(
    *,
    include_components: Optional[bool] = None,
    component_overrides: Optional[Mapping[str, bool]] = None,
    include_uncertainty: bool = True,
    quantile_subset: Optional[Iterable[float]] = None,
)
```

Eğitim geçmişine ait bileşen katkıları ve hata terimlerini döndürür. Parametre
anlamları `predict()` ile aynıdır ancak varsayılan görünürlük
`historical_components` yapılandırmasına bağlıdır.

### `backtest(...)`

```python
backtest(
    horizon: int,
    step: int = 1,
    *,
    strategy: Optional[str] = None,
    window: Optional[int] = None,
    include_components: Optional[bool] = None,
    component_overrides: Optional[Mapping[str, bool]] = None,
    include_uncertainty: bool = True,
    quantile_subset: Optional[Iterable[float]] = None,
)
```

* `strategy`: `BACKTEST_STRATEGIES` sabitinde listelenen yeniden eğitim
  yaklaşımlarından biri (`expanding`, `sliding`, `anchored`).
* `window`: `sliding` ve `anchored` stratejileri için pencere uzunluğunu
  belirler.
* Diğer parametreler `predict()` ile aynı davranışa sahiptir.

Çıktı veri çerçevesi temel doğruluk ölçütlerini (`mae`, `rmse`, `mape`, `r2`),
deneme aralığını (`start`, `end`), kullanılan stratejiyi ve eğitim boyutunu
içerir.

### `report() -> Dict[str, object]`

`ForecastReport` nesnesini sözlük olarak döndürür; performans metrikleri,
komponent güçleri ve OptiScorer tarzı aykırı değer özetlerini içerir.

## Veri kümesi yardımcıları

* `optitime.datasets.available_datasets()` – kayıtlı veri kümesi kimliklerini
  listeler.
* `optitime.datasets.dataset_info(name)` – veri kümesi açıklaması, frekans ve
  satır sayısı gibi meta bilgileri döndürür.
* `optitime.datasets.load_dataset(name)` – ilgili CSV dosyasını `pandas`
  veri çerçevesi olarak yükler.

## Kamuya açık sabitler

* `optitime.BACKTEST_STRATEGIES` – desteklenen backtest stratejileri dizisi.

Tüm API bileşenleri için kullanım örnekleri `README.md`, `tests/run_sales_example.py`
ve `tests/run_airlines_visuals.py` dosyalarında da gösterilmiştir.
