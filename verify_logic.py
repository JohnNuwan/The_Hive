
import sys
import os

# Add src/shared and src/eva-sentinel to path
sys.path.append(os.path.join(os.getcwd(), 'src', 'shared'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'eva-sentinel'))

try:
    from shared.internal_auth import InternalAuth
    print("✅ InternalAuth import success")
    
    # Test Generation/Verification
    token = InternalAuth.generate_token("test-node")
    payload = InternalAuth.verify_token(token)
    if payload and payload['src'] == "test-node":
        print("✅ JWT Generation/Verification: SUCCESS")
    else:
        print("❌ JWT Generation/Verification: FAILED")
        
except Exception as e:
    print(f"❌ InternalAuth Test Error: {e}")

try:
    from eva_sentinel.services.notifier import TelegramNotifier
    import asyncio
    
    async def test_notify():
        notifier = TelegramNotifier()
        print("✅ TelegramNotifier init success")
        # Test mock send
        res = await notifier.send_message("Verification Test")
        if res:
             print("✅ Notifier Mock send: SUCCESS")
        else:
             print("❌ Notifier Mock send: FAILED")
             
    asyncio.run(test_notify())
except Exception as e:
     print(f"❌ Notifier Test Error: {e}")
