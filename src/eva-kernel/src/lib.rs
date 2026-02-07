//! EVA Kernel — Bibliothèque de sécurité critique
//!
//! Ce crate expose les composants fondamentaux du Kernel :
//! - **Constitution** : Les 6 Lois Immuables
//! - **TradeValidator** : Validation Loi 2 (Protection du Capital)
//! - **KillSwitch** : Coupe-circuit d'urgence financier
//! - **AuditTrail** : Black Box blockchain-like (Loi 3)
//! - **Server** : Interface HTTP Axum
//! - **Protocols** : Dynasty (succession) + Phoenix (résurrection)

pub mod audit;
pub mod kill_switch;
pub mod laws;
pub mod protocols;
pub mod server;
pub mod validator;

pub use audit::AuditTrail;
pub use kill_switch::KillSwitch;
pub use laws::Constitution;
pub use server::start_kernel_server;
pub use validator::TradeValidator;
