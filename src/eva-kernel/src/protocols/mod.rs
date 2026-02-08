//! Protocols de survie du Kernel
//!
//! - **Dynasty** : Deadman Switch — transmission des accès aux héritiers
//! - **Phoenix** : Résurrection automatique via snapshots ZFS

pub mod dynasty;
pub mod phoenix;

// pub use dynasty::DynastyAccess;
pub use phoenix::PhoenixManager;
