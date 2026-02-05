/**
 * Dynasty Protocol - EVA KERNEL (Rust)
 * Gestion de la succession et accès révocable pour les héritiers.
 */

pub struct DynastyAccess {
    heir_public_key: String,
    inheritance_activated: bool,
}

impl DynastyAccess {
    pub fn new(heir_key: &str) -> Self {
        DynastyAccess {
            heir_public_key: heir_key.to_string(),
            inheritance_activated: false,
        }
    }

    pub fn verify_deadman_switch(&self, days_since_last_admin: i32) -> bool {
        if days_since_last_admin > 30 {
            println!("⌛ DYNASTY: Alerte Deadman Switch - Inactivité Admin détectée.");
            return true;
        }
        false
    }

    pub fn unlock_vault_for_heir(&mut self) {
        self.inheritance_activated = true;
        println!("👑 DYNASTY: Accès déchiffré pour l'héritier enregistré.");
    }
}
