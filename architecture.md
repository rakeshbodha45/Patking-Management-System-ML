
# System Architecture

```mermaid
flowchart TD
    A[Camera Input] --> B[OpenCV]
    B --> C[YOLO Machine Learning Model]
    C --> D[Vehicle Detection]
    D --> E[Parking Slot Detection]
    E --> F[Occupancy Calculation]
    F --> G[Available Slots]
    F --> H[Occupied Slots]
    G --> I[Display Results]
    H --> I
```

## Architecture Description

1. Camera captures the parking area.
2. OpenCV processes video frames.
3. YOLO detects vehicles.
4. Parking slots are checked.
5. Occupancy is calculated.
6. Available and occupied slots are displayed.
7. 
