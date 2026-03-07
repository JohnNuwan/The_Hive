import asyncio
import sys
import os

# Add src to path so we can import eva_muse
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "eva-muse"))
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "shared"))

from eva_muse.scheduler import MuseScheduler

async def main():
    print("🎬 Initialisation du Test de la Muse Factory...")
    scheduler = MuseScheduler()
    
    print("\n📸 Lancement du Test : Contenu STANDARD (Lifestyle/Trading)")
    await scheduler.generate_standard_content()
    
    print("\n🔥 Lancement du Test : Contenu NSFW (OnlyFans/Patreon)")
    await scheduler.generate_nsfw_content()
    
    print(f"\n✅ Tests terminés ! Vérifie le dossier : {scheduler.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
