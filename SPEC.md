# Lang2Robo — Demo MVP Specification

Платформа: текстовое описание бизнес-процесса → симуляция автоматизированной ячейки → итеративное улучшение. Без реального железа. Работает с любым типом малого бизнеса и любым набором оборудования — с роботами и без.

```
Фото помещения + текст сценария
  → 3D-реконструкция сцены
  → AI предлагает план роботизации (текст + схема)
  → Пользователь подтверждает
  → Автоскачивание моделей, SDK
  → Сборка прототипа в MuJoCo
  → Прогоны и итеративное улучшение политик
```

---

## Стек MVP

| Слой | Технология | Почему |
|------|-----------|--------|
| Симулятор | **MuJoCo** | `pip install mujoco`, CPU-only, 4000× realtime, Python API |
| Модели роботов | **MuJoCo Menagerie** + `robot_descriptions` | 135+ готовых MJCF/URDF, подбираются из knowledge-base |
| Каталог оборудования | **knowledge-base/equipment/** (JSON) | Claude подбирает только из реального каталога, не выдумывает |
| 3D-реконструкция | **DISCOVERSE** | Open-source Real2Sim пайплайн (фото → MuJoCo), MIT, 650 FPS рендер |
| Обучение политик | **LeRobot** + **SmolVLA** (450M) | Работает на MacBook, 30Hz, обучение в sim (LIBERO/Meta-World) |
| AI-планирование | **Claude API** | Vision для фото, текст для рекомендаций и итераций |
| Оркестрация | **asyncio** | Нулевые зависимости, линейный пайплайн |
| Бэкенд | **FastAPI** + **Pydantic** | API + валидация |
| Фронтенд | **React** + **TypeScript** + **Three.js** | 3D-превью сцены, редактор плана |
| Конвертация моделей | `urdf2mjcf`, `robot-format-converter` | URDF ↔ MJCF ↔ SDF |

**Минимальные требования**: Python 3.11+, 8 GB RAM, любой CPU. GPU не требуется.

---

## Модуль 1. Захват помещения

### Вход

Пользователь загружает 10–30 фото помещения (до 50 м²) через веб-интерфейс.

### Реконструкция — DISCOVERSE

**DISCOVERSE** (MIT, 2025) — Real2Sim фреймворк. Внутри: COLMAP (SfM) → 3D Gaussian Splatting → MJCF экспорт. Для пользователя — один вызов:

```
Фото (15-30 шт) → DISCOVERSE → MuJoCo сцена (MJCF + коллизионная геометрия)
```

Фотореалистичный рендер 650 FPS в MuJoCo.

Масштаб из фотограмметрии неизвестен (scale ambiguity). После реконструкции пользователь **отмечает два конца известного объекта** (дверь, стол, плитка) в Three.js 3D-превью и вводит реальный размер в метрах.

```python
class ReferenceCalibration(BaseModel):
    """Калибровка масштаба реконструкции."""

    point_a: tuple[float, float, float]  # Точка A в координатах меша
    point_b: tuple[float, float, float]  # Точка B в координатах меша
    real_distance_m: float               # Реальное расстояние между A и B

async def reconstruct_and_calibrate(
    photos_dir: Path,
    calibration: ReferenceCalibration,
) -> SceneReconstruction:
    """Реконструкция помещения + калибровка масштаба.

    Args:
        photos_dir: Директория с фото помещения.
        calibration: Reference-измерение от пользователя (две точки + реальный размер).

    Returns:
        Реконструированная сцена в реальном масштабе.
    """
    scene = discoverse.real2sim(
        image_path=photos_dir,
        output_path=photos_dir / "mujoco_scene",
    )

    mesh_distance = np.linalg.norm(
        np.array(calibration.point_a) - np.array(calibration.point_b)
    )
    scale_factor = calibration.real_distance_m / mesh_distance
    scene.apply_scale(scale_factor)

    return SceneReconstruction(
        mesh_path=scene.mesh_path,
        mjcf_path=scene.mjcf_path,
        pointcloud_path=scene.pointcloud_path,
        dimensions=scene.get_dimensions(),
    )
```

### Распознавание и структурирование — Claude Vision

Claude Vision анализирует фото **и** меш от DISCOVERSE, извлекая структурированные данные:
- Размеры помещения (из масштабированного меша DISCOVERSE)
- Существующее оборудование, зоны, двери, окна (из фото)

```python
async def analyze_scene(
    photos: list[Path],
    reconstruction: SceneReconstruction,
) -> SceneAnalysis:
    """Анализ помещения: фото + реконструкция → структурированные данные.

    Args:
        photos: Фото помещения.
        reconstruction: Результат DISCOVERSE (меш, MJCF).

    Returns:
        Размеры, зоны, существующее оборудование.
    """
    system_prompt = load_prompt("prompts/vision_analysis.md")

    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                *[image_content(photo) for photo in photos],
                text_content(f"Меш помещения: {reconstruction.dimensions}"),
            ],
        }],
    )
    return SceneAnalysis.model_validate_json(response.content[0].text)
```

### Объединение → SpaceModel

Реконструкция (DISCOVERSE MJCF) + анализ (Claude Vision) → `SpaceModel`:

```python
```python
class Dimensions(BaseModel):
    """Размеры помещения."""

    width_m: float
    length_m: float
    ceiling_m: float
    area_m2: float

class Zone(BaseModel):
    """Функциональная зона помещения."""

    name: str
    polygon: list[tuple[float, float]]  # 2D-контур в метрах
    area_m2: float

class Door(BaseModel):
    """Дверь в помещении."""

    position: tuple[float, float]
    width_m: float

class Window(BaseModel):
    """Окно в помещении."""

    position: tuple[float, float]
    width_m: float

class ExistingEquipment(BaseModel):
    """Оборудование, уже присутствующее в помещении."""

    name: str
    category: str
    position: tuple[float, float, float]
    confidence: float

class SceneReconstruction(BaseModel):
    """Результат DISCOVERSE Real2Sim."""

    mesh_path: Path
    mjcf_path: Path            # MJCF сцены помещения (база для добавления оборудования)
    pointcloud_path: Path
    dimensions: Dimensions     # Из масштабированного меша (после reference calibration)

class SceneAnalysis(BaseModel):
    """Результат анализа Claude Vision."""

    zones: list[Zone]
    existing_equipment: list[ExistingEquipment]
    doors: list[Door]
    windows: list[Window]

class SpaceModel(BaseModel):
    """Модель помещения для симуляции."""

    dimensions: Dimensions
    zones: list[Zone]
    existing_equipment: list[ExistingEquipment]
    doors: list[Door]
    windows: list[Window]
    reconstruction: SceneReconstruction  # Ссылка на MJCF и меш от DISCOVERSE
```
```

### Веб-интерфейс (этап 1)

1. Загрузка фото (drag-and-drop)
2. Ожидание реконструкции DISCOVERSE (прогресс-бар)
3. Three.js: 3D-превью меша. **Калибровка**: пользователь кликает две точки на известном объекте (дверь, стол) и вводит реальный размер в метрах
4. 2D floor plan сверху с распознанным оборудованием и зонами
5. Редактирование: корректировка зон, оборудования
6. Кнопка «Подтвердить план»

---

## Knowledge-base — каталог оборудования

Ключевой архитектурный элемент из оригинальной спеки. Claude **не выдумывает** оборудование — работает только по каталогу. Каждый `equipment_id` из ответа Claude валидируется по каталогу; если не найден — ретрай.

Только оборудование, которое можно симулировать в MuJoCo (есть MJCF/URDF-модель):

```
knowledge-base/
└── equipment/
    ├── manipulators.json    # SO-101, xArm, Franka, UR5e, Koch...
    ├── conveyors.json       # Модули конвейеров (с физикой ленты)
    ├── cameras.json         # Камеры (рендер в MuJoCo)
    └── fixtures.json        # Столы, полки, стойки, контейнеры (статическая геометрия)
```

Каждая позиция оборудования содержит:

```python
class EquipmentEntry(BaseModel):
    """Запись оборудования в каталоге."""

    id: str
    name: str
    type: Literal["manipulator", "conveyor", "camera", "fixture"]
    specs: dict                # reach, payload, размеры, скорость ленты и т.д.
    mjcf_source: MjcfSource    # menagerie_id или urdf_url
    price_usd: float | None = None
    purchase_url: str | None = None
    placement_rules: PlacementRules | None = None  # мин. зона, ограничения

class MjcfSource(BaseModel):
    """Источник MJCF-модели."""

    menagerie_id: str | None = None    # e.g. "franka_emika_panda"
    robot_descriptions_id: str | None = None
    urdf_url: str | None = None        # Прямая ссылка на URDF
```

> Датчики, актуаторы, контроллеры — не входят в MVP (нет MJCF-моделей, нужны только для реального железа).

Поле `type` определяет, как оборудование ведёт себя в симуляции:
- `manipulator` — управляется через IK/политику, выполняет pick/place/move
- `conveyor` — движет объекты по ленте, управляется через скорость
- `camera` — рендерит изображение для инспекции
- `fixture` — статическая геометрия (столы, полки), не управляется

---

## Модуль 2. AI-рекомендация

### Вход

SpaceModel + текстовое описание сценария автоматизации от пользователя.

Пример: *«Тёмная кухня, 3 рабочие станции. Манипулятор порционирует на станции 2. Конвейер перемещает контейнеры между станциями. Нужен контроль качества камерой.»*

### Процесс

Claude API получает:
- SpaceModel (JSON)
- Текст сценария
- **Каталог оборудования из knowledge-base**

```python
async def generate_recommendation(
    space: SpaceModel,
    scenario: str,
) -> Recommendation:
    """Генерация плана роботизации через Claude API.

    Args:
        space: Модель помещения.
        scenario: Текстовое описание сценария от пользователя.

    Returns:
        План роботизации с оборудованием из каталога.
    """
    catalog = load_equipment_catalog()
    system_prompt = load_prompt("prompts/recommendation.md")

    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": format_recommendation_context(
                space, scenario, catalog,
            ),
        }],
    )
    recommendation = parse_and_validate(response, catalog)
    return recommendation
```

**Валидация**: каждый `equipment_id` проверяется по каталогу. Цены берутся из каталога, не из ответа Claude. При невалидном id — ретрай (до 2 раз).

Claude возвращает **два формата**:

**1. Текстовый план** — человекочитаемое описание:
- Какое оборудование, почему, где стоит
- Последовательность действий
- Ожидаемые метрики

**2. Структурированный JSON** — машиночитаемый:

```python
class Recommendation(BaseModel):
    """План автоматизации помещения."""

    equipment: list[EquipmentPlacement]
    work_objects: list[WorkObject]       # Объекты для манипуляции (изделия, контейнеры)
    target_positions: dict[str, tuple[float, float, float]]  # Маппинг имя → координаты
    workflow_steps: list[WorkflowStep]
    expected_metrics: ExpectedMetrics

class EquipmentPlacement(BaseModel):
    """Размещение нового оборудования в сцене."""

    equipment_id: str      # ID из knowledge-base каталога
    position: tuple[float, float, float]
    orientation_deg: float
    purpose: str
    zone: str

class WorkObject(BaseModel):
    """Объект для манипуляции в симуляции (изделие, коробка, контейнер)."""

    name: str
    shape: Literal["box", "cylinder", "sphere"]
    size: tuple[float, float, float]  # Для box: x,y,z. Для cylinder: r,h,0
    mass_kg: float
    position: tuple[float, float, float]
    count: int = 1                     # Сколько экземпляров

class WorkflowStep(BaseModel):
    """Шаг рабочего процесса."""

    order: int
    action: str                    # "pick", "place", "move", "transport", "inspect", "wait"
    equipment_id: str | None = None  # None для "wait"
    target: str                    # Ключ из target_positions → 3D-координаты
    duration_s: float
    params: dict | None = None     # Доп. параметры (speed для конвейера и т.д.)
```

### Визуализация плана

Фронтенд отображает рекомендацию:
- Three.js: 3D-сцена с мешем помещения + контуры оборудования в позициях
- Текстовый план сбоку
- Кнопки: «Подтвердить» / «Изменить» (текстом, Claude переделает)

---

## Модуль 3. Автоскачивание и сборка сцены

### После подтверждения плана

Система автоматически:

1. **Скачивает модели оборудования** по данным из knowledge-base:
```python
async def download_equipment_models(
    placements: list[EquipmentPlacement],
) -> dict[str, Path]:
    """Скачивает MJCF/URDF-модели для всего оборудования из рекомендации.

    Args:
        placements: Список размещений из Recommendation.

    Returns:
        Маппинг equipment_id → путь к модели.
    """
    catalog = load_equipment_catalog()
    models = {}
    for p in placements:
        entry = catalog[p.equipment_id]
        if entry.menagerie_id:
            models[p.equipment_id] = get_menagerie_model(entry.menagerie_id)
        elif entry.urdf_url:
            models[p.equipment_id] = await download_urdf(entry.urdf_url)
    return models
```

2. **Собирает MJCF-сцену** — DISCOVERSE-меш как фон + интерактивные объекты:
```python
def generate_mjcf_scene(
    space: SpaceModel,
    recommendation: Recommendation,
    models: dict[str, Path],
    output_path: Path,
) -> Path:
    """Собирает финальную MJCF-сцену: помещение + оборудование + объекты.

    Args:
        space: Модель помещения (MJCF от DISCOVERSE + existing_equipment).
        recommendation: План автоматизации.
        models: Маппинг equipment_id → путь к MJCF-модели.
        output_path: Путь для сохранения собранной сцены.

    Returns:
        Путь к финальному MJCF-файлу.
    """
    base_mjcf = load_mjcf(space.reconstruction.mjcf_path)

    # Existing equipment как отдельные интерактивные тела
    # (упрощённые shapes поверх меша DISCOVERSE)
    for eq in space.existing_equipment:
        add_static_body(base_mjcf, name=eq.name, pos=eq.position,
                        shape="box", size=estimate_size(eq.category))

    # Новое оборудование из рекомендации
    for placement in recommendation.equipment:
        model_path = models[placement.equipment_id]
        add_equipment_to_scene(base_mjcf, model_path,
                               pos=placement.position,
                               orientation=placement.orientation_deg)

    # Рабочие объекты для манипуляции
    for obj in recommendation.work_objects:
        for i in range(obj.count):
            add_dynamic_body(base_mjcf, name=f"{obj.name}_{i}",
                             shape=obj.shape, size=obj.size,
                             mass=obj.mass_kg, pos=obj.position)

    save_mjcf(base_mjcf, output_path)
    return output_path
```

Три типа тел в сцене:
- **Фон** — меш DISCOVERSE (стены, пол, потолок, визуал)
- **Статические тела** — existing_equipment (принтеры, столы), не двигаются, но имеют коллизию
- **Динамические тела** — work_objects (изделия, коробки), можно хватать и перемещать

3. **Устанавливает зависимости** (если не установлены):
```bash
pip install mujoco mujoco-python-viewer lerobot robot_descriptions trimesh
```

---

## Модуль 4. Симуляция и прогоны

### Запуск

```python
async def run_simulation(
    scene_path: Path,
    workflow: list[WorkflowStep],
    catalog: dict[str, EquipmentEntry],
    target_positions: dict[str, tuple[float, float, float]],
    policy: LeRobotPolicy | None = None,  # None = scripted mode (MVP v1)
) -> SimResult:
    """Запускает симуляцию сцены в MuJoCo.

    Args:
        scene_path: Путь к MJCF-файлу сцены.
        workflow: Последовательность шагов рабочего процесса.
        catalog: Каталог оборудования (для определения типа).
        target_positions: Маппинг имя цели → 3D-координаты.
        policy: Обученная политика (MVP v2), None для scripted.

    Returns:
        Результаты симуляции с метриками.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    results = []
    for step in workflow:
        result = await execute_step(model, data, step, catalog, target_positions, policy)
        results.append(result)

    return SimResult(
        steps=results,
        metrics=compute_metrics(results),
    )
```

### Диспетчеризация по типу действия

Каждый `WorkflowStep` выполняется в зависимости от типа оборудования и действия:

```python
async def execute_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    catalog: dict[str, EquipmentEntry],
    target_positions: dict[str, tuple[float, float, float]],
    policy: LeRobotPolicy | None,
) -> StepResult:
    """Выполняет один шаг workflow в симуляции.

    Args:
        model: MuJoCo модель.
        data: MuJoCo данные симуляции.
        step: Шаг рабочего процесса.
        catalog: Каталог для определения типа оборудования.
        target_positions: Маппинг имя цели → 3D-координаты.
        policy: Обученная политика (опционально).

    Returns:
        Результат шага: успех, время, коллизии.
    """
    if step.action == "wait":
        return await sim_wait(model, data, step.duration_s)

    equipment_type = catalog[step.equipment_id].type

    if equipment_type == "manipulator":
        if step.action in ("pick", "place", "move"):
            if policy:
                return await learned_manipulation(model, data, step, policy)
            return await scripted_manipulation(model, data, step, target_positions)

    elif equipment_type == "conveyor":
        if step.action == "transport":
            return await sim_conveyor(model, data, step)

    elif equipment_type == "camera":
        if step.action == "inspect":
            return await sim_camera_inspect(model, data, step, target_positions)

    raise ValueError(f"Unknown action '{step.action}' for {equipment_type}")
```

### Исполнители по типам оборудования

**Манипулятор** — IK-контроллер (scripted) или обученная политика (learned):
```python
async def scripted_manipulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    target_positions: dict[str, tuple[float, float, float]],
) -> StepResult:
    """Манипуляция через IK-контроллер."""
    target_pos = target_positions[step.target]  # Резолвинг имени → координаты
    trajectory = compute_ik_trajectory(model, data, target_pos)
    return execute_trajectory(model, data, trajectory)

async def learned_manipulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    policy: LeRobotPolicy,
) -> StepResult:
    """Манипуляция через обученную политику (SmolVLA)."""
    obs = get_observation(model, data)
    while not is_done(model, data, step):
        action = policy.predict(obs)
        apply_action(model, data, action)
        mujoco.mj_step(model, data)
        obs = get_observation(model, data)
    return evaluate_result(model, data, step)
```

**Конвейер** — управление скоростью ленты:
```python
async def sim_conveyor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
) -> StepResult:
    """Симуляция конвейера: перемещение объектов по ленте."""
    conveyor_joint = find_joint(model, step.equipment_id)
    set_conveyor_speed(data, conveyor_joint, step.params.get("speed", 0.1))
    await sim_until(model, data, step.duration_s)
    return StepResult(success=True, duration_s=step.duration_s)
```

**Камера** — рендер + проверка видимости цели:
```python
async def sim_camera_inspect(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: WorkflowStep,
    target_positions: dict[str, tuple[float, float, float]],
) -> StepResult:
    """Симуляция инспекции: проверка что камера видит цель."""
    image = render_camera(model, data, camera_name=step.equipment_id)
    target_pos = target_positions[step.target]
    visible = is_in_camera_fov(model, data, step.equipment_id, target_pos)
    return StepResult(
        success=visible,
        duration_s=0.1,
        image=image,
        error=None if visible else f"Target '{step.target}' not in camera FOV",
    )
```

### Модели данных симуляции

```python
class StepResult(BaseModel):
    """Результат одного шага симуляции."""

    success: bool
    duration_s: float
    collision_count: int = 0
    error: str | None = None
    image: np.ndarray | None = None  # Для camera inspect

class SimResult(BaseModel):
    """Результат полного прогона симуляции."""

    steps: list[StepResult]
    metrics: SimMetrics

class SimMetrics(BaseModel):
    """Метрики прогона симуляции."""

    cycle_time_s: float
    success_rate: float       # 0.0–1.0
    collision_count: int
    failed_steps: list[int]   # Индексы неудавшихся шагов
```

### Визуализация

- **Встроенный MuJoCo viewer** — интерактивная 3D-визуализация прогона
- **Веб-интерфейс** — рендер кадров MuJoCo → WebSocket → Three.js (для удалённого просмотра)

---

## Модуль 5. Итеративное улучшение

### Цикл

```
Прогон → Метрики → Claude анализирует → Корректировки → Новый прогон
```

До 5 итераций. Claude получает:
- Текущие метрики (SimMetrics)
- Лог коллизий и ошибок
- Текущую конфигурацию сцены
- Историю предыдущих итераций

### Что корректирует Claude

1. **Позиции робота** — ближе/дальше к зоне pick/place
2. **Позиции объектов** — оптимизация рабочего пространства
3. **Параметры траектории** — высота подъёма, промежуточные waypoints
4. **Добавление/удаление объектов** — если не хватает стола, полки
5. **Смена модели робота** — если reach недостаточен, предложить другой

```python
```python
class PositionChange(BaseModel):
    """Изменение позиции оборудования."""

    equipment_id: str
    new_position: tuple[float, float, float]
    new_orientation_deg: float | None = None

class EquipmentReplacement(BaseModel):
    """Замена оборудования на другое из каталога."""

    old_equipment_id: str
    new_equipment_id: str   # Валидируется по каталогу
    reason: str

class SceneCorrections(BaseModel):
    """Коррекции от Claude после анализа метрик."""

    position_changes: list[PositionChange] | None = None
    add_equipment: list[EquipmentPlacement] | None = None
    remove_equipment: list[str] | None = None        # equipment_id
    replace_equipment: list[EquipmentReplacement] | None = None
    workflow_changes: list[WorkflowStep] | None = None  # Изменённые шаги

class IterationLog(BaseModel):
    """Лог одной итерации для контекста Claude."""

    iteration: int
    metrics: SimMetrics
    corrections_applied: SceneCorrections

async def iterate(
    scene_path: Path,
    metrics: SimMetrics,
    history: list[IterationLog],
    catalog: dict[str, EquipmentEntry],
) -> Path:
    """Одна итерация улучшения через Claude.

    Args:
        scene_path: Путь к текущему MJCF-файлу сцены.
        metrics: Метрики последнего прогона.
        history: Лог предыдущих итераций.
        catalog: Каталог оборудования (для валидации замен).

    Returns:
        Путь к скорректированному MJCF-файлу.
    """
    # Читаем MJCF-контент — Claude не может читать файлы
    scene_xml = scene_path.read_text()

    response = await claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        system=load_prompt("prompts/iteration.md"),
        messages=[{
            "role": "user",
            "content": format_iteration_context(
                scene_xml=scene_xml,
                metrics=metrics,
                history=history,
                catalog=catalog,
            ),
        }],
    )
    corrections = SceneCorrections.model_validate_json(response.content[0].text)

    # Если Claude предложил сменить оборудование — скачать новые модели
    if corrections.replace_equipment:
        for replacement in corrections.replace_equipment:
            validate_equipment_id(replacement.new_equipment_id, catalog)
            await download_equipment_model(replacement.new_equipment_id)

    new_scene_path = apply_corrections(scene_path, corrections)
    return new_scene_path
```
```

### Критерии остановки

- `success_rate >= 0.95` и `collision_count == 0` → успех
- 5 итераций без улучшения → остановка, отчёт пользователю
- Пользователь может остановить вручную

---

## Модуль 6. Обучение политик (MVP v2)

> Этот модуль применяется **только если в рекомендации есть манипуляторы**. Для сценариев без роботов (только конвейеры, камеры) — пайплайн завершается после Модуля 5.

### Пайплайн: scripted → демонстрации → обученная политика

**Шаг 1. Запись демонстраций** — scripted-контроллер (из Модуля 4, после успешных итераций Модуля 5) записывает траектории в формат LeRobot dataset:

```python
async def record_demonstrations(
    scene_path: Path,
    workflow: list[WorkflowStep],
    num_demos: int = 100,
    output_dir: Path = Path("data/projects/{id}/policies/demos"),
) -> Path:
    """Записывает успешные scripted-траектории как демонстрации.

    Args:
        scene_path: Финальная MJCF-сцена (после итераций).
        workflow: Шаги рабочего процесса.
        num_demos: Количество демонстраций (50-200).
        output_dir: Директория для LeRobot dataset.

    Returns:
        Путь к записанному dataset.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    dataset = LeRobotDataset(output_dir)

    for i in range(num_demos):
        data = mujoco.MjData(model)
        # Рандомизация начальных позиций объектов для разнообразия
        randomize_object_positions(data)

        obs_sequence = []
        for step in workflow:
            trajectory = compute_ik_trajectory(model, data, step)
            for action in trajectory:
                apply_action(model, data, action)
                mujoco.mj_step(model, data)
                obs_sequence.append(Observation(
                    image=render_camera(model, data),  # RGB с камеры сцены
                    joints=get_joint_positions(data),
                    action=action,
                ))

        dataset.add_episode(obs_sequence)

    dataset.save()
    return output_dir
```

**Шаг 2. Обучение SmolVLA** — fine-tune на записанных демонстрациях:
```bash
python -m lerobot.scripts.train \
  --policy.type=smolvla \
  --dataset.repo_id=local:workspace_demos \
  --env.type=mujoco \
  --env.task=pick_and_place_custom
```

**Шаг 3. Оценка** — прогон обученной политики в той же MuJoCo-сцене, сравнение метрик со scripted.

**Шаг 4. Итерация** — если метрики ниже scripted, добавить демонстраций (больше рандомизации) и переобучить.

### SmolVLA

- 450M параметров — работает на CPU/MacBook
- 30 Hz async inference
- Предобучена на данных LeRobot community
- Fine-tune на 50–200 демонстрациях для конкретной задачи
- **Observations**: RGB-изображение с камеры сцены + joint positions

---

## Пайплайн MVP — от начала до конца

```
┌─────────────────────────────────────┐
│  1. Пользователь                    │
│     • Загружает фото помещения      │
│     • Пишет сценарий текстом        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  2. Захват (Модуль 1)               │
│     • DISCOVERSE → MJCF помещения   │
│     • Claude Vision → зоны, оборуд. │
│     • → SpaceModel                  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  3. Рекомендация (Модуль 2)         │
│     • Claude API + каталог → JSON   │
│     • Визуализация в Three.js       │
│     • Пользователь подтверждает     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  4. Сборка (Модуль 3)               │
│     • Скачивание MJCF из каталога   │
│     • MJCF помещения + роботы       │
│     • → финальная MJCF-сцена        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  5. Симуляция (Модуль 4)            │
│     • MuJoCo: scripted IK           │
│     • Метрики: время, успех, коллизии│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  6. Итерации (Модуль 5)        ◄──┐ │
│     • Claude корректирует сцену   │ │
│     • Скачивание новых моделей    │ │
│     •   если смена оборудования   │ │
│     • Повторный прогон ───────────┘ │
│     • success_rate ≥ 0.95 → готово  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  7. Обучение (Модуль 6, MVP v2)     │
│     • Только если есть манипуляторы │
│     • Запись демонстраций (scripted) │
│     • LeRobot + SmolVLA fine-tune   │
│     • Оценка learned vs scripted    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  8. Результат                       │
│     • Финальная MJCF-сцена          │
│     • Обученная политика (v2)       │
│     • Отчёт с метриками             │
│     • Видео лучшего прогона         │
└─────────────────────────────────────┘
```

---

## Структура проекта

```
lang2robo/
├── pyproject.toml
├── docker-compose.yml        # API + Web
├── .env.example
│
├── knowledge-base/
│   └── equipment/            # JSON-каталог оборудования (манипуляторы, конвейеры, датчики...)
│
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entrypoint
│   │   ├── api/
│   │   │   ├── capture.py    # POST /capture — загрузка фото, анализ
│   │   │   ├── recommend.py  # POST /recommend — AI-рекомендация
│   │   │   ├── simulate.py   # POST /simulate — запуск симуляции
│   │   │   └── iterate.py    # POST /iterate — итерация улучшения
│   │   ├── models/           # Pydantic-модели
│   │   │   ├── space.py      # SpaceModel, Zone, Equipment
│   │   │   ├── recommendation.py  # Recommendation, RobotPlacement
│   │   │   └── simulation.py      # SimResult, SimMetrics
│   │   ├── services/
│   │   │   ├── vision.py     # Claude Vision анализ фото
│   │   │   ├── planner.py    # Claude рекомендация + итерации
│   │   │   ├── scene.py      # Генерация MJCF-сцены
│   │   │   ├── simulator.py  # MuJoCo прогоны
│   │   │   └── downloader.py # Скачивание моделей из Menagerie
│   │   └── core/
│   │       ├── config.py     # Settings (Pydantic)
│   │       ├── claude.py     # Claude API клиент
│   │       └── prompts.py    # load_prompt() — загрузка промптов из prompts/
│   └── tests/
│       ├── test_vision.py
│       ├── test_planner.py
│       ├── test_scene.py
│       └── test_simulator.py
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── PhotoUpload.tsx
│   │   │   ├── FloorPlanEditor.tsx
│   │   │   ├── SceneViewer3D.tsx    # Three.js MuJoCo сцена
│   │   │   ├── RecommendationView.tsx
│   │   │   ├── SimulationPlayer.tsx
│   │   │   └── MetricsDashboard.tsx
│   │   └── types/
│   │       └── index.ts
│   └── tsconfig.json
│
├── prompts/
│   ├── vision_analysis.md     # System prompt: анализ фото помещения
│   ├── recommendation.md      # System prompt: генерация плана роботизации
│   └── iteration.md           # System prompt: коррекция сцены по метрикам
│
├── models/                    # Кэш скачанных MJCF-моделей
│
├── data/
│   └── projects/
│       └── {project_id}/
│           ├── photos/            # Исходные фото пользователя
│           ├── reconstruction/    # Выход DISCOVERSE (меш, point cloud, MJCF)
│           ├── recommendation/    # JSON-рекомендация от Claude
│           ├── scenes/            # MJCF-сцены по итерациям (v1.xml, v2.xml...)
│           ├── simulations/       # Метрики + видео прогонов
│           ├── policies/          # Обученные политики + демонстрации (MVP v2)
│           └── report.json        # Финальный отчёт
│
└── scripts/
    ├── train_policy.py        # LeRobot fine-tune (MVP v2)
    └── record_demos.py        # Запись демонстраций из scripted (MVP v2)
```

---

## Зависимости

### Backend (Python)

```toml
[project]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]",
    "pydantic>=2.0",
    "anthropic>=0.40",
    "mujoco>=3.0",
    "robot-descriptions",
    "discoverse",
    "trimesh",
    "numpy",
    "pillow",
]

[project.optional-dependencies]
training = [
    "lerobot",
    "torch",
]
```

### Frontend (TypeScript)

```json
{
  "dependencies": {
    "react": "^19",
    "three": "^0.170",
    "@react-three/fiber": "^9",
    "@react-three/drei": "^10"
  }
}
```

---

## Переменные окружения

```env
ANTHROPIC_API_KEY=sk-ant-...   # Единственный внешний сервис
```

---

## Запуск (Demo MVP)

```bash
git clone https://github.com/user/lang2robo
cd lang2robo
pip install -e ".[training]"
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app --reload
# Открыть http://localhost:8000
```

---

## Примеры сценариев

### Сценарий 1: Студия 3D-печати (робот + конвейер + камера)

**Помещение**: 30 м², 5 принтеров Bambu Lab, рабочий стол, стеллаж.

**Сценарий пользователя**: *«Робот снимает готовые изделия с build plate принтеров, ставит на конвейер. Конвейер перемещает к столу постобработки. Камера детектирует сбои печати.»*

**Recommendation от Claude**:
```
equipment: [franka_emika_panda, conveyor_500mm, camera_overhead]

work_objects: [
  WorkObject("finished_print", shape="box", size=(0.05, 0.05, 0.04), mass_kg=0.1,
             position=(1.0, 1.0, 0.85), count=5)
]

target_positions: {
  "printer_1_bed": (1.0, 1.0, 0.85),
  "printer_2_bed": (2.5, 1.0, 0.85),
  "printer_3_bed": (4.0, 1.0, 0.85),
  "conveyor_start": (3.0, 2.0, 0.85),
  "conveyor_end":   (3.0, 4.0, 0.85),
  "post_table":     (3.0, 4.5, 0.85),
}

workflow_steps: [
  (1, "inspect",   camera_overhead,    "printer_1_bed"),
  (2, "pick",      franka_emika_panda, "printer_1_bed"),
  (3, "place",     franka_emika_panda, "conveyor_start"),
  (4, "transport", conveyor_500mm,     "conveyor_end", params={"speed": 0.05}),
  (5, "wait",      None,               "next_print_ready"),
]
```

**MJCF-сцена**: фон DISCOVERSE + 5 принтеров (static bodies) + Franka + конвейер + камера + 5 изделий (dynamic bodies).

**Симуляция**: inspect → pick → place → transport → wait. Все шаги диспатчатся по типу оборудования.

**Итерация**: Franka не дотянулась до printer_3 → Claude сдвигает робота на (2.5, 1.5, 0.0) → повтор → success.

**Обучение (MVP v2)**: Есть манипулятор → 100 демонстраций с рандомизацией позиций изделий → SmolVLA fine-tune.

---

### Сценарий 2: ПВЗ — пункт выдачи заказов (БЕЗ робота)

**Помещение**: 20 м², стеллаж, окно выдачи, стол приёмки.

**Сценарий пользователя**: *«Пункт выдачи. Конвейер перемещает посылки от стола приёмки к стеллажу. Камера сканирует штрих-коды для сортировки. Без робота.»*

**Recommendation от Claude**:
```
equipment: [conveyor_1000mm, camera_barcode]

work_objects: [
  WorkObject("parcel_small", shape="box", size=(0.20, 0.15, 0.10), mass_kg=0.5,
             position=(1.0, 1.0, 0.85), count=10),
  WorkObject("parcel_large", shape="box", size=(0.40, 0.30, 0.20), mass_kg=2.0,
             position=(1.0, 1.5, 0.85), count=5),
]

target_positions: {
  "reception_table": (1.0, 1.0, 0.85),
  "conveyor_start":  (1.5, 1.0, 0.85),
  "conveyor_end":    (4.0, 1.0, 0.85),
  "shelf_zone":      (4.5, 1.0, 0.85),
}

workflow_steps: [
  (1, "inspect",   camera_barcode,  "reception_table"),
  (2, "transport", conveyor_1000mm, "conveyor_end", params={"speed": 0.08}),
  (3, "wait",      None,            "next_parcel"),
]
```

**MJCF-сцена**: фон DISCOVERSE + стеллаж/стол (static bodies) + конвейер + камера + 15 посылок двух размеров (dynamic bodies).

**Симуляция**: inspect → transport → wait. Нет pick/place — нет IK. Конвейер двигает посылки через трение MuJoCo.

**Итерация**: Камера не видит reception_table (плохой угол). Claude корректирует позицию камеры → повтор → visible=true.

**Обучение**: Нет манипуляторов → **Модуль 6 пропускается**.

---

### Сценарий 3: Мастерская ремонта электроники (робот + камера, без конвейера)

**Помещение**: 15 м², паяльная станция, микроскоп, стол с компонентами.

**Сценарий пользователя**: *«Мастерская ремонта. Робот берёт плату со стола приёмки, подносит к камере-микроскопу для инспекции, перемещает на паяльную станцию.»*

**Recommendation от Claude**:
```
equipment: [koch_v1_1, camera_microscope]

work_objects: [
  WorkObject("pcb_board", shape="box", size=(0.10, 0.07, 0.002), mass_kg=0.05,
             position=(0.5, 0.5, 0.75), count=3)
]

target_positions: {
  "intake_table":      (0.5, 0.5, 0.75),
  "microscope_fov":    (1.5, 0.5, 0.80),
  "soldering_station": (2.5, 0.5, 0.75),
}

workflow_steps: [
  (1, "pick",    koch_v1_1,         "intake_table"),
  (2, "move",    koch_v1_1,         "microscope_fov"),
  (3, "inspect", camera_microscope, "microscope_fov"),
  (4, "place",   koch_v1_1,         "soldering_station"),
]
```

**MJCF-сцена**: фон DISCOVERSE + паяльная станция/микроскоп (static bodies) + Koch v1.1 + камера + 3 платы (dynamic bodies, mass=0.05 кг).

**Симуляция**: pick → move (удерживая плату) → inspect → place. Koch — маленький arm для точных операций.

**Итерация**: Koch не дотягивается до soldering_station (reach 0.28 м, нужно 0.35 м). Claude предлагает `replace_equipment`: Koch → Franka. Система скачивает Franka из Menagerie, пересобирает сцену → повтор → success.

**Обучение (MVP v2)**: Есть манипулятор → 100 демонстраций с рандомизацией позиций плат → SmolVLA fine-tune.

---

### Сценарий 4: Тёмная кухня (робот + конвейер + камера)

**Помещение**: 35 м², плита, холодильник, 3 рабочие станции.

**Сценарий пользователя**: *«Тёмная кухня. Манипулятор порционирует на станции 2. Конвейер подаёт контейнеры от станции 1 к станции 3. Камера контролирует порционирование.»*

**Recommendation от Claude**:
```
equipment: [franka_emika_panda, conveyor_500mm, camera_overhead]

work_objects: [
  WorkObject("food_container", shape="box", size=(0.15, 0.10, 0.08), mass_kg=0.3,
             position=(1.0, 1.5, 0.85), count=5)
]

target_positions: {
  "station_1":       (1.0, 1.5, 0.85),
  "station_2":       (3.0, 1.5, 0.85),
  "station_3":       (5.0, 1.5, 0.85),
  "conveyor_start":  (1.0, 2.0, 0.85),
  "conveyor_end":    (5.0, 2.0, 0.85),
}

workflow_steps: [
  (1, "pick",      franka_emika_panda, "station_1"),
  (2, "place",     franka_emika_panda, "station_2"),
  (3, "wait",      None,               "portioning_done"),
  (4, "inspect",   camera_overhead,    "station_2"),
  (5, "pick",      franka_emika_panda, "station_2"),
  (6, "place",     franka_emika_panda, "conveyor_start"),
  (7, "transport", conveyor_500mm,     "conveyor_end", params={"speed": 0.05}),
]
```

**MJCF-сцена**: фон DISCOVERSE + плита/холодильник/столы (static bodies) + Franka + конвейер + камера + 5 контейнеров (dynamic bodies).

**Симуляция**: pick → place → wait → inspect → pick → place → transport. Полный цикл с манипуляцией и конвейером.

**Итерация**: Коллизия arm ↔ стол при step 6. Claude сдвигает Franka выше (z += 0.1) → повтор → success.

**Обучение (MVP v2)**: Есть манипулятор → 100 демонстраций → SmolVLA fine-tune.
