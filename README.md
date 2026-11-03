# Neuralized LeGO-LOAM for Edge AI

**Neuralization of a LOAM-Based LiDAR-SLAM Pipeline under Edge-AI Constraints**

---

# English

## 1. Overview

This project introduces a neural ground segmentation module into the LeGO-LOAM LiDAR-SLAM pipeline, replacing the original ground removal approach with a Convolutional Neural Network (CNN).

This project is based on the [FishROS](https://github.com/fishros/LeGO-LOAM-ROS2) fork of the original LeGO-LOAM project, which has been extended with a CNN-based ground segmentation module for Edge-AI deployment.

The neural network processes LiDAR range images and predicts ground segmentation labels. Its output is integrated into the original ROS 2 pipeline.

The main objective is to investigate the feasibility of neural ground segmentation under Edge-AI constraints, with particular attention to deployment on the NVIDIA Jetson Orin Nano platform.

## 2. Pipeline Architecture

The modified pipeline combines the original LeGO-LOAM framework with an ONNX-based neural inference module.

The main processing stages are:

1. LiDAR point cloud acquisition.
2. Range image projection.
3. Neural ground segmentation.
4. Ground and non-ground point classification.
5. Point cloud segmentation and feature extraction.
6. LiDAR odometry and mapping.

The original ground removal method is retained as an alternative to neural segmentation.

Two operating modes are available:

- **Classical mode:** uses the original LeGO-LOAM ground removal approach.
- **Neural mode:** uses the CNN model through ONNX Runtime.

## 3. Neural Ground Segmentation

The CNN receives a LiDAR range image with the following dimensions:

- **Input:** `16 × 1800`
- **Output:** ground segmentation labels.

The trained network is deployed in ONNX format and integrated into the C++ ROS 2 application through ONNX Runtime.

The repository includes the inference model:

```text
neural_network/
└── model/
    ├── ground_segmentation.onnx
    └── ground_segmentation.onnx.data
```

Both files are required by the distributed model.

The model is automatically installed into the ROS 2 package share directory during compilation. Its path is resolved by the launch file, without requiring a machine-specific absolute path.

## 4. Repository Structure

```text
lego-loam-edge-ai/
├── algorithm/
│   ├── LeGO-LOAM/
│   │   ├── config/
│   │   ├── include/
│   │   ├── launch/
│   │   ├── rviz/
│   │   ├── src/
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   └── cloud_msgs/
│       ├── msg/
│       ├── CMakeLists.txt
│       └── package.xml
│
├── neural_network/
│   ├── model/
│   │   ├── ground_segmentation.onnx
│   │   └── ground_segmentation.onnx.data
│   └── README.md
│
├── docs/
├── results/
└── README.md
```

## 5. Requirements

The project has been tested in the following environment:

- Ubuntu Linux
- ROS 2 Humble
- C++14 compatible compiler
- PCL
- Eigen3
- GTSAM
- Boost
- ONNX Runtime

The target deployment platform is NVIDIA Jetson Orin Nano.

**Important:** ONNX Runtime must match the target hardware architecture. The x86-64 version can be used for development and testing on a desktop computer, while Jetson deployment requires a compatible ARM64 build.

## 6. Installation and Build

### 6.1 Clone the repository

Create a ROS 2 workspace:

```bash
mkdir -p ~/lego_loam_ws/src
cd ~/lego_loam_ws/src
```

Clone the repository:

```bash
git clone <REPOSITORY_URL>
```

The repository contains both ROS 2 packages:

- `lego_loam_sr`
- `cloud_msgs`

Colcon discovers both packages inside the cloned repository.

### 6.2 Configure ONNX Runtime

Install or extract a compatible ONNX Runtime distribution.

It is possible to install the required version here: https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2

Set the environment variable to the ONNX Runtime installation directory, which must contain the `include` and `lib` directories:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

For example, on a desktop development environment:

```bash
export ONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.19.2
```

### 6.3 Build the workspace

Source ROS 2:

```bash
source /opt/ros/humble/setup.bash
```

Build the required packages:

```bash
cd ~/lego_loam_ws

colcon build --packages-up-to lego_loam_sr
```

Source the workspace:

```bash
source install/setup.bash
```

During compilation, the ONNX model is automatically installed into:

```text
install/lego_loam_sr/share/lego_loam_sr/model/
```

No manual model path configuration is required for the standard launch configuration.

## 7. Dataset and Execution

The ROS 2 bag dataset is distributed separately from the source code.

You can download the ROS2 dataset here: https://drive.google.com/drive/folders/1ZyZgvbxI4RlcKWgf7nDuWA3HYYzTuiA8?usp=sharing

### 7.1 Start LeGO-LOAM

In a terminal, source the ROS 2 environment and the workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/lego_loam_ws/install/setup.bash
```

Start the pipeline using the classical ground removal method:

```bash
ros2 launch lego_loam_sr run.launch.py
```

### 7.2 Enable neural ground segmentation

To activate CNN-based ground segmentation:

```bash
ros2 launch lego_loam_sr run.launch.py use_neural_ground:=true
```

The ONNX model is located automatically through the installed ROS 2 package.

### 7.3 Play the dataset

In a separate terminal:

```bash
source /opt/ros/humble/setup.bash

ros2 bag play /path/to/bag
```

Make sure the dataset publishes the expected LiDAR topic and that the ROS 2 environment is configured correctly.

The launch configuration includes the following topic remapping:

```text
/rslidar_points -> /velodyne_points
```

## 8. Launch Parameters

The main launch file supports the following parameters:

| Parameter | Default | Description |
|---|---|---|
| `use_neural_ground` | `false` | Enables neural ground segmentation |
| `diagnostics` | `false` | Enables diagnostic mode |
| `rviz` | `true` | Enables RViz visualization |
| `enable_end_to_end_timing` | `false` | Enables end-to-end latency measurements |

Example:

```bash
ros2 launch lego_loam_sr run.launch.py \
    rviz:=true \
    use_neural_ground:=true \
    diagnostics:=false \
    enable_end_to_end_timing:=false
```

## 9. Deployment

The project investigates the integration of neural ground segmentation into a resource-constrained LiDAR-SLAM pipeline.

Inference is performed directly within the C++ ROS 2 application using ONNX Runtime.

The target platform is NVIDIA Jetson Orin Nano.

---

# Italiano

## 1. Panoramica

Questo progetto introduce un modulo neurale per la segmentazione del terreno nella pipeline LiDAR-SLAM LeGO-LOAM, sostituendo il metodo originale di rimozione del terreno con una rete neurale convoluzionale (CNN).

Questo progetto si basa sulla fork [FishROS](https://github.com/fishros/LeGO-LOAM-ROS2) del progetto originale LeGO-LOAM ed è stato successivamente esteso con un modulo di segmentazione del terreno basato su CNN, progettato per il deployment in ambienti Edge-AI.

La rete neurale elabora immagini di range LiDAR e predice le etichette di segmentazione del terreno. I risultati vengono successivamente integrati nella pipeline ROS 2 originale.

L'obiettivo principale è studiare la fattibilità della segmentazione neurale del terreno in condizioni di vincoli Edge-AI, con particolare attenzione al deployment sulla piattaforma NVIDIA Jetson Orin Nano.

## 2. Architettura della pipeline

La pipeline modificata combina il framework originale LeGO-LOAM con un modulo di inferenza neurale basato su ONNX.

Le principali fasi di elaborazione sono:

1. Acquisizione della point cloud LiDAR.
2. Proiezione della point cloud in una range image.
3. Segmentazione neurale del terreno.
4. Classificazione dei punti appartenenti e non appartenenti al terreno.
5. Segmentazione della point cloud ed estrazione delle feature.
6. Odometria LiDAR e mapping.

Il metodo originale di rimozione del terreno viene mantenuto come alternativa alla segmentazione neurale.

Sono disponibili due modalità operative:

- **Modalità classica:** utilizza il metodo originale di rimozione del terreno di LeGO-LOAM.
- **Modalità neurale:** utilizza il modello CNN attraverso ONNX Runtime.

## 3. Segmentazione neurale del terreno

La CNN riceve in ingresso un'immagine di range LiDAR con le seguenti dimensioni:

- **Input:** `16 × 1800`
- **Output:** etichette di segmentazione del terreno.

La rete addestrata viene distribuita in formato ONNX e integrata nell'applicazione C++ ROS2 attraverso ONNX Runtime.

La repository contiene il modello utilizzato per l'inferenza:

```text
neural_network/
└── model/
    ├── ground_segmentation.onnx
    └── ground_segmentation.onnx.data
```

Entrambi i file sono necessari per il modello distribuito.

Durante la compilazione, il modello viene installato automaticamente nella directory condivisa del package ROS 2. Il percorso viene individuato dal launch file, senza richiedere percorsi assoluti specifici della macchina.

## 4. Struttura della repository

```text
lego-loam-edge-ai/
├── algorithm/
│   ├── LeGO-LOAM/
│   │   ├── config/
│   │   ├── include/
│   │   ├── launch/
│   │   ├── rviz/
│   │   ├── src/
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   └── cloud_msgs/
│       ├── msg/
│       ├── CMakeLists.txt
│       └── package.xml
│
├── neural_network/
│   ├── model/
│   │   ├── ground_segmentation.onnx
│   │   └── ground_segmentation.onnx.data
│   └── README.md
│
├── docs/
├── results/
└── README.md
```

## 5. Requisiti

Il progetto è stato testato nel seguente ambiente:

- Ubuntu Linux
- ROS 2 Humble
- Compilatore compatibile con C++14
- PCL
- Eigen3
- GTSAM
- Boost
- ONNX Runtime

La piattaforma di riferimento per il deployment è NVIDIA Jetson Orin Nano.

**Importante:** ONNX Runtime deve essere compatibile con l'architettura hardware di destinazione. La versione x86-64 può essere utilizzata per lo sviluppo e i test su un computer desktop, mentre il deployment su Jetson richiede una versione compatibile ARM64.

## 6. Installazione e compilazione

### 6.1 Clonazione della repository

Creare un workspace ROS 2:

```bash
mkdir -p ~/lego_loam_ws/src
cd ~/lego_loam_ws/src
```

Clonare la repository:

```bash
git clone <REPOSITORY_URL>
```

La repository contiene entrambi i package ROS 2:

- `lego_loam_sr`
- `cloud_msgs`

Colcon individua automaticamente entrambi i package all'interno della repository clonata.

### 6.2 Configurazione di ONNX Runtime

Installare o estrarre una distribuzione compatibile di ONNX Runtime.

È possibile installare la versione richiesta qui: https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2

Impostare la variabile d'ambiente sul percorso della directory di installazione di ONNX Runtime, che deve contenere le cartelle `include` e `lib`:

```bash
export ONNXRUNTIME_ROOT=/percorso/di/onnxruntime
```

Ad esempio, in un ambiente desktop:

```bash
export ONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.19.2
```

### 6.3 Compilazione del workspace

Caricare l'ambiente ROS 2:

```bash
source /opt/ros/humble/setup.bash
```

Compilare i package richiesti:

```bash
cd ~/lego_loam_ws

colcon build --packages-up-to lego_loam_sr
```

Caricare il workspace:

```bash
source install/setup.bash
```

Durante la compilazione, il modello ONNX viene installato automaticamente in:

```text
install/lego_loam_sr/share/lego_loam_sr/model/
```

Non è necessaria alcuna configurazione manuale del percorso del modello per il launch standard.

## 7. Dataset ed esecuzione

Il dataset ROS 2 bag viene distribuito separatamente dal codice sorgente.

È possibile scaricare il dataset ROS2 qui: https://drive.google.com/drive/folders/1ZyZgvbxI4RlcKWgf7nDuWA3HYYzTuiA8?usp=sharing

### 7.1 Avvio di LeGO-LOAM

In un terminale, caricare l'ambiente ROS 2 e il workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/lego_loam_ws/install/setup.bash
```

Avviare la pipeline utilizzando il metodo classico di rimozione del terreno:

```bash
ros2 launch lego_loam_sr run.launch.py
```

### 7.2 Attivazione della segmentazione neurale

Per attivare la segmentazione neurale basata sulla CNN, lanciare in un terminale:

```bash
ros2 launch lego_loam_sr run.launch.py use_neural_ground:=true
```

Il modello ONNX viene individuato automaticamente attraverso il package ROS 2 installato.

### 7.3 Riproduzione del dataset

In un terminale separato:

```bash
source /opt/ros/humble/setup.bash

ros2 bag play /percorso/del/bag
```

Assicurarsi che il dataset pubblichi il topic LiDAR previsto e che l'ambiente ROS 2 sia configurato correttamente.

La configurazione di launch include il seguente remapping:

```text
/rslidar_points -> /velodyne_points
```

## 8. Parametri di avvio

Il launch file principale supporta i seguenti parametri:

| Parametro | Valore predefinito | Descrizione |
|---|---|---|
| `rviz` | `true` | Attiva la visualizzazione RViz |
| `use_neural_ground` | `false` | Attiva la segmentazione neurale |
| `diagnostics` | `false` | Attiva la modalità diagnostica |
| `enable_end_to_end_timing` | `false` | Attiva la misurazione delle latenze end-to-end |

Esempio:

```bash
ros2 launch lego_loam_sr run.launch.py \
    rviz:=true \
    use_neural_ground:=true \
    diagnostics:=false \
    enable_end_to_end_timing:=false
```

## 9. Deployment

Il progetto studia l'integrazione della segmentazione neurale del terreno in una pipeline LiDAR-SLAM con risorse computazionali limitate.

L'inferenza viene eseguita direttamente nell'applicazione C++ ROS 2 utilizzando ONNX Runtime.

La piattaforma di riferimento per il deployment è NVIDIA Jetson Orin Nano.