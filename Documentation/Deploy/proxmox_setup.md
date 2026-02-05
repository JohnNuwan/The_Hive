# Guide d'Installation Proxmox VE - THE HIVE

> **Version**: 1.0.0  
> **Durée estimée**: 2-3 heures  
> **Prérequis**: Serveur physique avec GPU dédié

---

## 📋 Prérequis Matériels

### Minimum (Genesis Phase)
- **CPU**: Intel/AMD 8+ cores (Ryzen 7 ou équivalent)
- **RAM**: 128 GB DDR4/DDR5
- **GPU**: NVIDIA RTX 3090 24GB (ou équivalent)
- **Stock**: NVMe 1TB + HDD 4TB pour backup
- **Réseau**: 2x NIC (1 WAN, 1 LAN)

### Hardware Sécurité
- **The Tablet**: Clé USB industrielle 4-8GB avec switch physique
- **The Vault**: YubiKey 5 NFC ou Nitrokey Pro 2
- **The Watchdog**: ESP32 DevKit (optionnel Phase 1)

---

## 🔧 Étape 1: Installation Proxmox VE

### 1.1 Téléchargement
```bash
# Télécharger l'ISO Proxmox VE 8.x
wget https://www.proxmox.com/en/downloads
# Vérifier le SHA256
sha256sum proxmox-ve_8.x-x.iso
```

### 1.2 Création USB Bootable
```bash
# Linux
sudo dd if=proxmox-ve_8.x.iso of=/dev/sdX bs=4M status=progress

# Windows: Utiliser Rufus en mode DD
```

### 1.3 Installation
1. Booter sur la clé USB
2. Sélectionner le disque NVMe pour l'OS
3. Configurer:
   - **Hostname**: `proxmox-hive`
   - **IP**: `10.0.0.1/24` (ou selon votre réseau)
   - **Gateway**: Votre routeur
   - **DNS**: `1.1.1.1` ou local
4. Définir le mot de passe root et email admin
5. Laisser l'installation se terminer et rebooter

### 1.4 Post-Installation
```bash
# Se connecter en SSH
ssh root@10.0.0.1

# Désactiver le repo enterprise (sauf si licence)
sed -i 's/^deb/#deb/' /etc/apt/sources.list.d/pve-enterprise.list

# Ajouter le repo no-subscription
echo "deb http://download.proxmox.com/debian/pve bookworm pve-no-subscription" > /etc/apt/sources.list.d/pve-no-sub.list

# Mettre à jour
apt update && apt full-upgrade -y
```

---

## 🌐 Étape 2: Configuration Réseau

### 2.1 Bridges Virtuels

Éditer `/etc/network/interfaces`:

```bash
# WAN Bridge (DMZ)
auto vmbr0
iface vmbr0 inet static
    address 192.168.1.100/24
    gateway 192.168.1.1
    bridge-ports enp1s0
    bridge-stp off
    bridge-fd 0

# Internal Bridge (VMs)
auto vmbr1
iface vmbr1 inet static
    address 10.0.1.1/24
    bridge-ports none
    bridge-stp off
    bridge-fd 0
    post-up   echo 1 > /proc/sys/net/ipv4/ip_forward
    post-up   iptables -t nat -A POSTROUTING -s '10.0.1.0/24' -o vmbr0 -j MASQUERADE
    post-down iptables -t nat -D POSTROUTING -s '10.0.1.0/24' -o vmbr0 -j MASQUERADE
```

```bash
# Appliquer
systemctl restart networking
```

### 2.2 Firewall Proxmox

Via l'interface web (https://10.0.0.1:8006):
1. Datacenter → Firewall → Options → Enable
2. Ajouter règles:
   - `DROP` tout par défaut
   - `ACCEPT` SSH depuis Tailscale
   - `ACCEPT` HTTPS (8006) depuis Tailscale

---

## 🎮 Étape 3: GPU Passthrough (IOMMU)

### 3.1 Activer IOMMU

```bash
# Éditer GRUB
nano /etc/default/grub

# Ajouter à GRUB_CMDLINE_LINUX_DEFAULT (Intel):
GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt"

# Ou pour AMD:
GRUB_CMDLINE_LINUX_DEFAULT="quiet amd_iommu=on iommu=pt"

# Appliquer
update-grub
```

### 3.2 Modules VFIO

```bash
# Éditer modules
nano /etc/modules

# Ajouter:
vfio
vfio_iommu_type1
vfio_pci
vfio_virqfd

# Identifier l'ID GPU
lspci -nn | grep NVIDIA
# Ex: 01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA102 [GeForce RTX 3090] [10de:2204]

# Bloquer drivers natifs
echo "options vfio-pci ids=10de:2204,10de:1aef disable_vga=1" > /etc/modprobe.d/vfio.conf
echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
echo "blacklist nvidia" >> /etc/modprobe.d/blacklist.conf

# Reconstruire initramfs
update-initramfs -u -k all

# Reboot
reboot
```

### 3.3 Vérification

```bash
# Vérifier IOMMU
dmesg | grep -e DMAR -e IOMMU

# Vérifier groupe IOMMU du GPU
find /sys/kernel/iommu_groups/ -type l | grep "01:00"
```

---

## 🖥️ Étape 4: Création des VMs

### 4.1 Template Ubuntu 22.04 (EVA Core & Sentinel)

```bash
# Télécharger cloud-init image
wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

# Créer VM template
qm create 9000 --name "ubuntu-22.04-template" --memory 2048 --net0 virtio,bridge=vmbr1

# Importer disque
qm importdisk 9000 jammy-server-cloudimg-amd64.img local-lvm

# Configurer
qm set 9000 --scsihw virtio-scsi-pci --scsi0 local-lvm:vm-9000-disk-0
qm set 9000 --ide2 local-lvm:cloudinit
qm set 9000 --boot c --bootdisk scsi0
qm set 9000 --serial0 socket --vga serial0
qm set 9000 --agent 1

# Convertir en template
qm template 9000
```

### 4.2 VM EVA-Core

```bash
# Cloner depuis template
qm clone 9000 100 --name "eva-core" --full

# Configurer ressources
qm set 100 --memory 65536 --cores 8
qm set 100 --net0 virtio,bridge=vmbr1

# GPU Passthrough
qm set 100 --hostpci0 01:00,pcie=1,x-vga=1

# Cloud-init
qm set 100 --ciuser eva --cipassword "CHANGE_ME" --ipconfig0 ip=10.0.1.100/24,gw=10.0.1.1

# Resize disk
qm resize 100 scsi0 +200G

# Démarrer
qm start 100
```

### 4.3 VM Trading Floor (Windows 11)

1. Télécharger ISO Windows 11 + VirtIO drivers
2. Créer VM via l'interface web:
   - **ID**: 200
   - **Name**: trading-floor
   - **Memory**: 32GB
   - **CPU**: 4 cores
   - **Disk**: 200GB (local-lvm)
   - **Network**: vmbr1
3. Installer Windows 11
4. Installer VirtIO drivers
5. Configurer IP statique: `10.0.1.200/24`

### 4.4 VM Sentinel

```bash
# Cloner depuis template
qm clone 9000 150 --name "sentinel" --full

# Configurer ressources
qm set 150 --memory 16384 --cores 4
qm set 150 --net0 virtio,bridge=vmbr1

# Cloud-init
qm set 150 --ciuser sentinel --cipassword "CHANGE_ME" --ipconfig0 ip=10.0.1.150/24,gw=10.0.1.1

# Resize disk
qm resize 150 scsi0 +100G

# Démarrer
qm start 150
```

---

## 🔐 Étape 5: Configuration The Tablet

### 5.1 Préparer la Clé USB

```bash
# Identifier la clé
lsblk

# Formater en ext4
mkfs.ext4 /dev/sdX1 -L THE_LAW

# Monter temporairement
mkdir -p /mnt/tablet-temp
mount /dev/sdX1 /mnt/tablet-temp

# Copier les fichiers Constitution
cp /path/to/Lois.toml /mnt/tablet-temp/Constitution.toml
cp /path/to/kernel.sha512 /mnt/tablet-temp/

# Démonter
umount /mnt/tablet-temp
```

### 5.2 Configuration Auto-Mount (Read-Only)

```bash
# Éditer fstab
nano /etc/fstab

# Ajouter:
LABEL=THE_LAW /mnt/THE_LAW ext4 ro,noexec,nosuid,nodev 0 0

# Créer point de montage
mkdir -p /mnt/THE_LAW
mount /mnt/THE_LAW
```

### 5.3 Activer la Protection Physique

1. Démonter la clé
2. Activer le switch "Lock" sur le boîtier USB
3. Remonter - la clé sera hardware read-only

---

## 🔒 Étape 6: Tailscale VPN

### 6.1 Installation sur Proxmox

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --ssh --hostname=proxmox-hive
```

### 6.2 Installation sur chaque VM

```bash
# Ubuntu
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --hostname=eva-core
```

---

## ✅ Étape 7: Vérification Finale

### Checklist

- [ ] Proxmox accessible via https://TAILSCALE_IP:8006
- [ ] GPU visible dans VM eva-core (`nvidia-smi`)
- [ ] Réseau 10.0.1.0/24 fonctionnel entre VMs
- [ ] The Tablet montée en /mnt/THE_LAW (read-only)
- [ ] Toutes les VMs accessibles via Tailscale
- [ ] Firewall Proxmox actif et configuré

### Tests de Connectivité

```bash
# Depuis eva-core
ping 10.0.1.200  # Trading VM
ping 10.0.1.150  # Sentinel VM
nvidia-smi       # GPU disponible

# Depuis trading-floor
ping 10.0.1.100  # Core VM
```

---

## 📝 Prochaines Étapes

1. **Installer Docker** sur eva-core et sentinel
2. **Déployer les services** via docker-compose
3. **Configurer Wazuh** sur sentinel
4. **Installer MT5** sur trading-floor
5. **Tester la communication Redis** entre VMs

---

> [!TIP]
> Sauvegardez régulièrement vos VMs avec `vzdump`:
> ```bash
> vzdump 100 --storage local --mode snapshot
> ```
