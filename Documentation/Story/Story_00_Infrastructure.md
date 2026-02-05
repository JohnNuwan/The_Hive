# Story 00: Infrastructure & DevOps (Foundation)

## 📌 Context
Before running any AI, we need a stable, secure, and resilient infrastructure. The "Genesis" phase relies on a single server (Proxmox) but must behave like a professional datacenter.

## 🎯 Objectives
- Install and configure Proxmox VE.
- Secure the network (Segmentation).
- Prepare the environment for AI (GPU Passthrough) and Trading (Windows VM).

## 📋 Epic/Tasks Breakdown

### TASK-00-01: Proxmox Setup (The Hive Metal)
- **Role**: SysAdmin
- **Description**: Bare-metal installation of Proxmox VE 8.x.
- **Acceptance Criteria**:
    - [ ] Proxmox accessible via HTTPS (IP LAN).
    - [ ] Storage ZFS configured (even effectively single disk, for snapshot capabilities).
    - [ ] "The Watchdog" Hardware wiring prepared (Script ready to listen to GPIO if using RPi, or just the concept documented).

### TASK-00-02: Network Topology (The Nervous System)
- **Role**: Network Engineer
- **Description**: Create internal bridges.
- **Specs**:
    - `vmbr0`: Bridge WAN (Access to Router).
    - `vmbr1` (Intranet): 10.0.1.x/24 - High speed inter-VM communication (VirtIO).
    - `vmbr2` (DMZ): 10.0.2.x/24 - For "The Arena" (Isolated).
- **Acceptance Criteria**:
    - [ ] A VM in `vmbr0` can ping internet.
    - [ ] A VM in `vmbr1` cannot be accessed from WAN directly.

### TASK-00-03: VM Boilerplate Generation
- **Role**: DevOps
- **Description**: Create reusable templates (Cloud-Init).
- **Deliverables**:
    - `Template-Ubuntu-AI`: Ubuntu 22.04 LTS + Python 3.10 + Nvidia Drivers pre-installed.
    - `Template-Win10-Trading`: Windows 10 LTSC stripped of bloatware.
- **Acceptance Criteria**:
    - [ ] Spawning a new VM takes < 2 minutes.

### TASK-00-04: GPU Isolation (The 3090)
- **Role**: SysAdmin
- **Description**: Enable IOMMU and bind RTX 3090 to vfio-pci.
- **Acceptance Criteria**:
    - [ ] `lspci -nnk` shows the GPU bound to vfio.
    - [ ] VM 100 ("The Brain") detects the card (`nvidia-smi` works inside VM).

### TASK-00-05: Git & CI/CD Init
- **Role**: Lead Dev
- **Description**: Initialize the monorepo structure.
- **Acceptance Criteria**:
    - [ ] Repo created.
    - [ ] Pre-commit hooks installed (Ruff, Black).
    - [ ] Dockerfiles for Core and Banker created.
