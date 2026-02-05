import logging
import asyncio
from uuid import uuid4
from eva_core.services.memory import get_memory_service
from shared import ChatMessage, MessageRole

# Activer le logging pour voir Mem0 en action
logging.basicConfig(level=logging.INFO)

async def test_adaptive_learning():
    """
    Simule une interaction où EVA apprend une préférence utilisateur.
    """
    print("--- 🧠 TEST DE MÉMOIRE ADAPTATIVE (Mem0) ---")
    
    memory_service = get_memory_service()
    
    # 1. On simule un message où l'utilisateur exprime une préférence
    pref_message = ChatMessage(
        session_id=uuid4(),
        role=MessageRole.USER,
        content="Je m'appelle John et je préfère ne jamais trader le Gold après 20h car c'est trop volatil."
    )
    
    print(f"\n[USER] : {pref_message.content}")
    print("Action : EVA stocke le message et Mem0 extrait les faits...")
    
    # Stockage (déclenche Mem0 en interne dans MemoryService.store_message)
    await memory_service.store_message(pref_message)
    
    # 2. On récupère le profil 'appris' par la Ruche
    print("\nAction : Récupération du profil utilisateur appris...")
    profile = memory_service.get_user_profile()
    
    print("\n--- 📝 PROFIL APPRIS (Mem0) ---")
    if not profile:
        print("Aucun fait extrait (Mem0 est peut-être en mode mock ou n'a pas trouvé de fait saillant).")
    else:
        for p in profile:
            print(f"- {p}")
            
    # 3. On simule une question pour voir si la mémoire épisodique (recherche vectorielle) fonctionne aussi
    print("\nAction : Recherche sémantique sur 'Gold'...")
    search_results = await memory_service.search("Gold", limit=1)
    
    if search_results:
        print(f"Trouvé en mémoire Qdrant : '{search_results[0]['content']}' (Score: {search_results[0]['score']:.2f})")

if __name__ == "__main__":
    asyncio.run(test_adaptive_learning())
