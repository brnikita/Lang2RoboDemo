# Plan: Presentation-Quality Simulation Engine

## Context

Lang2Robo — универсальный генератор автоматизации. Пользователь описывает **любой** бизнес-процесс текстом, Claude подбирает оборудование из каталога, строит сцену, симулирует, оптимизирует. Это НЕ конкретный сценарий — это платформа.

Примеры из SPEC:
- 3D print farm (робот + конвейер + камера)
- Пункт выдачи (конвейер + камера, БЕЗ робота)
- Ремонт электроники (робот + камера, БЕЗ конвейера)
- Dark kitchen (робот + конвейер + камера)

**Текущая проблема**: Все оборудование в MuJoCo отображается как цветные кубики. Нет артикулированных роботов, конвейеры не двигают объекты, IK-контроллер только проверяет дистанцию. Клиенту нечего показать — он видит стопку кубиков вместо работающего автоматизированного процесса.

**Цель**: Сделать симуляцию визуально презентабельной для ЛЮБОГО сценария, который генерирует Claude.

---

## Gap Analysis: Current vs SPEC

### Module 3 — Scene Assembly

| SPEC | Current | Gap |
|---|---|---|
| Download MJCF from Menagerie → include в сцену | Модели скачиваются, но **не включаются** — вместо них box geom | **Критический**: нужно `<include>` реальных MJCF |
| `add_equipment_to_scene(mjcf, model_path, pos, orientation)` | Генерирует `<body><geom type="box"/></body>` | Не использует `model_path` |
| Work objects — dynamic bodies | `<freejoint>` + box geom | ОК, но нет grasp/attachment |
| Existing equipment — static bodies | Box geom, brown color | Приемлемо для MVP |
| DISCOVERSE mesh как background | Не интегрирован | Отдельная задача (Module 1) |

### Module 4 — Simulation Executors

| SPEC: Equipment Type | SPEC: Behavior | Current | Gap |
|---|---|---|---|
| **manipulator** (pick/place/move) | `compute_ik_trajectory()` → `execute_trajectory()` | Distance check → success/fail | Нет IK, нет движения суставов, нет grasp |
| **conveyor** (transport) | `find_joint()` → `set_conveyor_speed()` → `sim_until()` | `mj_step()` without any force | Объекты не двигаются |
| **camera** (inspect) | `render_camera()` → `is_in_camera_fov()` | Geometric FOV check only | Рендеринг не реализован |
| **wait** | Physics stepping | `mj_step()` | ОК |
| **learned policy** (MVP v2) | `policy.predict(obs)` → `apply_action()` | Не реализовано | Фаза 2 |

### Module 5 — Iteration

| SPEC | Current | Gap |
|---|---|---|
| Claude корректирует позиции → re-simulation | XML edits → re-simulation | Работает, но re-simulation не показывает улучшений визуально потому что робот не двигается |
| Equipment replacement → download new model | Скачивание работает | Новая модель тоже становится кубиком |

---

## Architecture: Features по слоям

Разделяю по слоям системы, а не по конкретным механикам. Каждая фича — универсальная для всех сценариев.

### Feature 1: Real Equipment Models in Scenes (Module 3)

**Что**: При сборке сцены — включать реальные MJCF-модели оборудования вместо box geoms.

**Покрывает**:
- Манипуляторы (xarm7, franka, ur5e, aloha, kinova, sawyer, widow_x, so_arm100) — полные mesh + joints + actuators из MuJoCo Menagerie
- Конвейеры — параметрический MJCF с slide joint + actuator (нет в Menagerie, генерируем)
- Камеры — `<camera>` element (уже есть) + визуальный корпус body
- Fixtures — box geoms (приемлемо, столы/полки не требуют mesh)

**Как**:
- Для `type == "manipulator"`: Use MuJoCo `<include file="..."/>` с relative path к скачанной модели. Menagerie MJCF самодостаточны — MuJoCo резолвит mesh paths relative to included file.
- Для `type == "conveyor"`: Генерировать параметрический MJCF — body с belt surface geom + slide joint + velocity actuator. Размеры из каталога (length_m, width_m).
- Для `type == "camera"`: Оставить `<camera>` element + добавить small box geom для визуализации корпуса.
- Для `type == "fixture"`: Оставить box geom (текущий подход).

**Файлы**:
- `backend/app/services/scene.py` — основные изменения
- `backend/app/services/downloader.py` — убедиться что модели реально скачиваются

**Commit**: `feat: include real MJCF models for all equipment types`

---

### Feature 2: Universal Action Executors (Module 4)

**Что**: Реализовать настоящие executors для каждого `action` type из workflow.

**Покрывает ВСЕ сценарии**, т.к. workflow всегда состоит из комбинации:

#### 2a: Manipulator actions (pick / place / move)

Для **любого** манипулятора из каталога (xarm7, franka, ur5e, и т.д.):

- **IK controller**: Jacobian transpose method через `mj_jacSite()`. Универсальный — работает с любым количеством joints/DOF. End-effector определяется через `<site>` на gripper (все Menagerie модели имеют end-effector sites).
- **Grasp**: MuJoCo weld equality constraint — attach object body к gripper site. Runtime on/off через `model.eq_active`.
- **pick** = IK to target → activate weld → IK to lift height
- **place** = IK to drop position → deactivate weld → object falls via gravity
- **move** = IK to target (with or without grasped object)

#### 2b: Conveyor actions (transport)

Для **любого** конвейера из каталога (500mm, 1000mm, 2000mm):

- Belt velocity через actuator control: `data.ctrl[actuator_id] = speed`
- Объекты перемещаются за счёт friction с belt surface
- Duration определяет как долго belt работает

#### 2c: Camera actions (inspect)

Для **любой** камеры (overhead, microscope, barcode):

- FOV check (текущий, работает)
- Optional: `mujoco.Renderer` для actual frame capture → сохранение image

#### 2d: Wait action

- Physics stepping (текущий, работает)

**Файлы**:
- `backend/app/services/simulator.py` — основные изменения
- Новый модуль `backend/app/services/controllers.py` — IK controller, grasp controller (переиспользуемые для любого манипулятора)

**Commit**: `feat: universal action executors for all equipment types`

---

### Feature 3: Visual Simulation Mode (Module 4)

**Что**: "Launch Viewer" запускает MuJoCo viewer и проигрывает **весь workflow** визуально — робот двигается, конвейер крутится, объекты перемещаются.

**Покрывает**: Все сценарии. Клиент нажимает кнопку — видит полный цикл автоматизации.

**Как**:
- `mujoco.viewer.launch_passive()` — non-blocking viewer
- Loop: для каждого workflow step → execute controller → viewer renders
- Viewer sync через `viewer.sync()` на каждом physics step

**Файлы**:
- `backend/app/services/simulator.py` — добавить `run_visual_simulation()`
- `backend/app/api/simulate.py` — `POST /{id}/view` вызывает visual runner

**Commit**: `feat: visual simulation runner with real-time MuJoCo viewer`

---

### Feature 4: Iteration Visibility (Module 5)

**Что**: После "Run Optimization" — итеративный loop уже работает (Claude корректирует → re-sim). Нужно чтобы каждая итерация **визуально отличалась** — робот перемещен, workflow изменен, метрики улучшились.

**Покрывает**: Все сценарии. Оптимизация универсальна:
- Позиции оборудования (Claude двигает робот ближе к цели)
- Замена оборудования (reach не хватает → Claude меняет робот на другой из каталога)
- Изменение workflow (убрать лишний step, добавить промежуточную точку)

**Что нужно**:
- С Feature 1-3 итерации уже будут визуально значимыми (робот реально двигается → видно что изменилось)
- Добавить: при замене оборудования → скачать новую модель → пересобрать сцену с реальным MJCF
- Добавить: видео-запись каждой итерации для сравнения (optional: `mujoco.Renderer` → frame → mp4)

**Файлы**:
- `backend/app/services/iteration.py` — при `replace_equipment` пересобирать сцену с новыми моделями
- Опционально: `backend/app/services/recorder.py` — video recording

**Commit**: `feat: iteration loop with real model replacement and visual diff`

---

## Implementation Order

```
Feature 1: Real models in scenes         ← Prerequisite for everything
    ↓
Feature 2: Universal action executors     ← Makes simulation actually work
    ↓
Feature 3: Visual simulation mode         ← Client-facing demo
    ↓
Feature 4: Iteration visibility           ← Optimization demo
```

Каждая фича — отдельный коммит с тестами. После Feature 3 уже можно демонстрировать клиенту.

---

### Feature 5: Point Cloud Quality (Module 1 — Capture)

**Что**: Облако точек отображается неправильно — разреженное, перевёрнутое, висит под сценой. На скриншоте: ~1000 точек вместо десятков тысяч, Y-ось инвертирована.

**Три корневые проблемы**:

#### 5a: Координатная система — COLMAP vs Three.js

COLMAP (OpenCV convention): X-right, **Y-down**, Z-forward.
Three.js: X-right, **Y-up**, Z-toward-viewer.

Текущий код (`reconstruction.py:198`): `points.append(point3d.xyz)` — никакой трансформации. Frontend (`SceneViewer3D.tsx`) читает координаты as-is. Результат: облако точек отрисовывается **перевёрнутым** — комната "висит" под сеткой.

**Решение**: В `_export_pointcloud()` трансформировать координаты:
```python
# COLMAP (X, Y, Z) → Three.js (X, -Z, Y)  или  (X, -Y, -Z)
transformed = np.column_stack([points[:, 0], -points[:, 1], -points[:, 2]])
```

#### 5b: Point cloud не пересчитывается при калибровке

`calibrate_scale()` масштабирует mesh (OBJ) и MJCF, но **не облако точек** (PLY). После калибровки размеры на экране обновляются (1.24m × 1.15m), но point cloud в Three.js остаётся в старых координатах (12.4m × 11.5m). Визуально — облако огромное, а сетка маленькая.

**Решение**: В `calibrate_scale()` также пересчитать PLY:
```python
def _rescale_pointcloud(ply_path: Path, scale_factor: float) -> None:
    cloud = trimesh.load(ply_path)
    cloud.vertices *= scale_factor
    cloud.export(ply_path)
```

#### 5c: Разреженность — нужна dense reconstruction

Текущий pipeline: COLMAP SfM → sparse points (300-1100 вершин). Это **структурные точки** из feature matching — их всегда мало.

Для presentation quality нужна **dense reconstruction**: COLMAP MVS (Multi-View Stereo) или аналог. Это даёт 50K-500K точек — комната визуально узнаваема.

**Решение (два варианта)**:

**Вариант A — pycolmap dense stereo** (если доступен):
```python
mvs = pycolmap.PatchMatchStereo()
mvs.run(output_dir / "dense", reconstruction)
# → depth maps → dense point cloud
```

**Вариант B — увеличить sparse quality** (быстрый фикс):
- `max_num_features = 32768` (вместо 8192)
- `max_ratio = 0.8` (строже, но больше точных матчей)
- `min_num_matches = 15` (фильтрация шума)
- Добавить `sequential_matching` для смежных фото (улучшает покрытие)

**Вариант B рекомендуется для MVP** — даёт 3-5x больше точек без major refactoring.

#### 5d: Grid не соответствует масштабу

Grid в Three.js: `gridHelper args={[10, 20]}` — фиксированный 10×10. Если point cloud 12m wide — он выходит за сетку. Если 1.2m — сетка в 10 раз больше.

**Решение**: Динамический grid — размер = bounding box point cloud × 1.5.

**Файлы**:
- `backend/app/services/reconstruction.py` — coordinate transform, rescale PLY, quality settings
- `backend/app/api/capture.py` — rescale PLY during calibration
- `frontend/src/components/SceneViewer3D.tsx` — dynamic grid size

**Commit**: `fix: point cloud coordinate transform, calibration rescale, dynamic grid`

---

## Updated Implementation Order

```
Feature 5: Point cloud quality fixes      ← Quick wins, immediate visual improvement
    ↓
Feature 1: Real models in scenes         ← Prerequisite for simulation
    ↓
Feature 2: Universal action executors     ← Makes simulation actually work
    ↓
Feature 3: Visual simulation mode         ← Client-facing demo
    ↓
Feature 4: Iteration visibility           ← Optimization demo
```

Feature 5 идёт **первой** потому что это быстрые фиксы (координаты, масштаб, grid) с немедленным визуальным эффектом на этапе Calibrate.

---

## Available Resources

### Robot Models (robot_descriptions 1.23.0)
57 MuJoCo-ready моделей. Все из каталога имеют MJCF:
- `xarm7_mj_description` → `ufactory_xarm7/xarm7.xml`
- `panda_mj_description` → `franka_emika_panda/panda.xml`
- `ur5e_mj_description` → `universal_robots_ur5e/ur5e.xml`
- `aloha_mj_description` → `aloha/scene.xml`
- `kinova_gen3_mj_description` → `kinova_gen3/gen3.xml`

### MuJoCo 3.6.0 Capabilities
- `mj_jacSite()` — Jacobian для IK (любое количество DOF)
- `mj_step()` — physics at 500Hz
- `eq_active` — runtime weld constraints для grasp
- `data.ctrl[]` — actuator control для conveyor velocity
- `mujoco.viewer.launch_passive()` — non-blocking viewer
- `mujoco.Renderer` — offscreen rendering для camera/video

### What's NOT Available (and not needed for MVP)
- DISCOVERSE — room reconstruction (Module 1, separate task)
- LeRobot + SmolVLA — policy training (Module 6, MVP v2)

---

## Expected Demo Flow (after implementation)

1. Клиент загружает фото → получает 3D point cloud
2. Описывает сценарий текстом (любой бизнес-процесс)
3. Claude генерирует план — список оборудования из каталога
4. "Confirm & Build Scene" → скачиваются реальные MJCF модели
5. **MuJoCo viewer показывает**: articulated robot arm (mesh), conveyor belt, work objects, camera
6. **"Run Simulation"** → робот двигается к цели, захватывает объект, ставит на конвейер, конвейер везёт объект, камера проверяет
7. Метрики: cycle time, success rate, collisions
8. **"Run Optimization"** → Claude корректирует → робот перемещён → re-simulation с улучшенными метриками
9. Итог: финальная сцена + метрики + опционально видео
