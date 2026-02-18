use std::time::Duration;
use tokio::time::{sleep, interval};

#[tokio::test]
async fn test_starvation_repro() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);

    // Spawn a sender task that floods the channel
    tokio::spawn(async move {
        loop {
            if tx.send("msg").await.is_err() {
                break;
            }
            // Send messages faster than the timeout (e.g., every 1ms vs 20ms timeout)
            sleep(Duration::from_millis(1)).await;
        }
    });

    let mut timeout_triggered_count = 0;
    let start = std::time::Instant::now();
    // Run for 200ms. With 20ms timeout, we expect ~10 timeouts if working correctly.
    // In starvation case, we expect 0 or very few.
    let max_duration = Duration::from_millis(200);

    loop {
        if start.elapsed() > max_duration {
            break;
        }

        tokio::select! {
            _ = rx.recv() => {
                // Process message simulating work
                // sleep(Duration::from_micros(100)).await;
            }
            _ = sleep(Duration::from_millis(20)) => {
                timeout_triggered_count += 1;
            }
        }
    }

    println!("Starvation Repro: Timeout triggered {} times", timeout_triggered_count);
    // In the starvation case, timeout should NOT trigger because sleep is reset
    assert!(timeout_triggered_count == 0, "Timeout triggered unexpectedly in starvation test! Count: {}", timeout_triggered_count);
}

#[tokio::test]
async fn test_fix_verification() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);

    // Spawn a sender task that floods the channel
    tokio::spawn(async move {
        loop {
            if tx.send("msg").await.is_err() {
                break;
            }
            sleep(Duration::from_millis(1)).await;
        }
    });

    let mut timeout_triggered_count = 0;
    let start = std::time::Instant::now();
    let max_duration = Duration::from_millis(200);

    // FIX: Define interval outside the loop
    let mut interval = interval(Duration::from_millis(20));

    // Consume the first tick which is immediate
    interval.tick().await;

    loop {
        if start.elapsed() > max_duration {
            break;
        }

        tokio::select! {
            _ = rx.recv() => {
                // Process message
            }
            _ = interval.tick() => {
                timeout_triggered_count += 1;
            }
        }
    }

    println!("Fix Verification: Timeout triggered {} times", timeout_triggered_count);
    // With the fix, timeout SHOULD trigger roughly 200ms / 20ms = 10 times.
    // Allowing some slack.
    assert!(timeout_triggered_count >= 8, "Timeout failed to trigger enough times in fixed implementation! Count: {}", timeout_triggered_count);
}
