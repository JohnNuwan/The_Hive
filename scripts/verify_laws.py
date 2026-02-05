import hashlib
import os
import sys

def calculate_sha512(file_path):
    """Calcule le hash SHA-512 d'un fichier."""
    sha512_hash = hashlib.sha512()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha512_hash.update(byte_block)
    return sha512_hash.hexdigest()

def verify_constitution(tablet_path="/mnt/THE_LAW", config_file="Lois.toml"):
    """
    Vérifie l'intégrité de la Constitution contre son hash enregistré.
    """
    target = os.path.join(tablet_path, config_file)
    
    if not os.path.exists(target):
        print(f"ERROR : Tablette non montee ou {config_file} manquant.")
        sys.exit(1)
        
    # Tentative de lecture simple si toml manque
    try:
        import toml
        config = toml.load(target)
        expected_hash = config.get("metadata", {}).get("constitution_hash", "")
    except ImportError:
        print("INFO : Module 'toml' non detecte. Verification du fichier uniquement.")
        expected_hash = ""
        
    try:
        if not expected_hash:
            print("INFO : Aucun hash trouve ou 'toml' manquant. Phase Genesis ?")
            current_hash = calculate_sha512(target)
            print(f"Hash calcule pour signature : {current_hash}")
            return
            
        current_hash = calculate_sha512(target)
        
        if current_hash == expected_hash:
            print("OK : Integrite validee.")
        else:
            print("ALERTE : La Constitution a ete modifiee !")
            print(f"Attendu : {expected_hash}")
            print(f"Actuel  : {current_hash}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR : Erreur technique : {e}")
        sys.exit(1)

if __name__ == "__main__":
    local_test_path = "Documentation/Config"
    verify_constitution(tablet_path=local_test_path, config_file="lois_template.toml")
