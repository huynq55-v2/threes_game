use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use threes_rs::game::Direction;
use threes_rs::n_tuple_network::NTupleNetwork;
use threes_rs::threes_env::ThreesEnv;

fn main() {
    // 1. Setup Thread Pool (8 threads)
    let num_threads = 8;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!(
        "🚀 Starting Multi-threaded Player ({} threads)",
        num_threads
    );

    // 2. Load Brain
    let args: Vec<String> = env::args().collect();
    let brain_path = if args.len() > 1 {
        args[1].clone()
    } else {
        match find_latest_checkpoint() {
            Some(ep) => format!("brain_ep_{}.msgpack", ep),
            None => {
                eprintln!("❌ No brain file found! Please specify one or run train.rs first.");
                return;
            }
        }
    };

    println!("📂 Loading brain from: {}", brain_path);
    let brain = NTupleNetwork::load_from_msgpack(&brain_path).expect("Failed to load brain");
    let shared_brain = Arc::new(brain);

    // 3. Configuration
    let total_games = 500;
    let gamma = 1.0; // Gamma doesn't matter for pure play, but env needs it.
    let epsilon = 0.0; // Pure greedy

    println!("🎮 Playing {} games...", total_games);
    let start_time = Instant::now();

    // 4. Parallel Execution
    // We use a Mutex to collect results safely, or we can just collect into a vector.
    // Vector collection is faster and cleaner with rayon.
    let results: Vec<(f64, u32)> = (0..total_games)
        .into_par_iter()
        .map(|i| {
            let mut env = ThreesEnv::new(gamma);
            // Clone reference to brain (cheap)
            let brain_ref = shared_brain.clone();
            let mut rng = rand::rng();

            let mut step = 0;
            // env.reset(); // new() already resets, but safe to call if needed logic changes

            // We need to access get_best_action_recursive which takes &self and &NTupleNetwork
            // It doesn't modify brain, so we can pass it directly.

            while !env.game.is_game_over() {
                // Epsilon greedy (e=0 -> always best)
                // We use 0.0 directly as requested

                let action = env.get_best_action_depth(&brain_ref, 3).0;

                env.game.make_full_move(action.unwrap());

                // if !moved {
                //     // Check game over strictly if move failed
                //     if !env.game.can_move(Direction::Up)
                //         && !env.game.can_move(Direction::Down)
                //         && !env.game.can_move(Direction::Left)
                //         && !env.game.can_move(Direction::Right)
                //     {
                //         env.game.game_over = true;
                //     }
                // } else {
                //     // Logic game step of env includes spawn
                //     // But here we are calling move_dir directly on game to control the loop better?
                //     // Wait, ThreesEnv::step does move_dir AND spawn.
                //     // Let's check ThreesEnv::step implementation in previous turn.
                //     // It returns (observation, reward, done, hints).
                //     // But get_best_action_recursive expects us to just play.
                //     // The standard way to play using Env is usually:
                //     // let (obs, reward, done, hints) = env.step(action);

                //     // Let's look at train.rs again.
                //     // train.rs uses:
                //     // local_env.train_step(&mut local_brain, action, current_alpha);
                //     // Inside train_step, it calls self.step(action).

                //     // usage in step():
                //     // if self.game.can_move(dir) { self.game.move_dir(dir); ... }

                //     // So we should use env.step(action) to ensure consistency (score update, etc)
                //     // BUT `step` does not return the structured info we might want (like max tile)
                //     // accessible via env.game.

                //     // However, `step` calculates reward which we don't need for inference.
                //     // We just need the state update.
                //     // Let's just use `env.step(action)`.
                //     // Wait, `step` returns `(Vec<u32>, f64, bool, Vec<u32>)`.
                // }

                step += 1;
                // Safety break
                if step > 20000 {
                    break;
                }
            }

            if i % 50 == 0 {
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }

            (env.game.score, env.game.get_highest_tile_value())
        })
        .collect();

    println!(
        "\n✅ Finished in {:.2}s",
        start_time.elapsed().as_secs_f64()
    );

    // 5. Statistics
    let mut max_score = 0.0;
    let mut total_score = 0.0;
    let mut tile_counts = HashMap::new();

    for (score, max_tile) in &results {
        if *score > max_score {
            max_score = *score;
        }
        total_score += *score;
        *tile_counts.entry(*max_tile).or_insert(0) += 1;
    }

    let avg_score = total_score / total_games as f64;

    println!("\n📊 STATISTICS ({} games)", total_games);
    println!("--------------------------------------------------");
    println!("Average Score: {:.2}", avg_score);
    println!("Max Score:     {:.0}", max_score);
    println!("--------------------------------------------------");
    println!("Max Tile Distribution:");

    let mut sorted_tiles: Vec<u32> = tile_counts.keys().cloned().collect();
    sorted_tiles.sort();

    for tile in sorted_tiles {
        let count = tile_counts.get(&tile).unwrap();
        let percentage = (*count as f64 / total_games as f64) * 100.0;
        println!("  {:>5}: {:>3} games ({:.1}%)", tile, count, percentage);
    }
    println!("--------------------------------------------------");
}

// Copied from train.rs
fn find_latest_checkpoint() -> Option<u32> {
    let mut max_ep = 0;
    let mut found = false;

    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("brain_ep_") && name.ends_with(".msgpack") {
                    let num_part = name
                        .trim_start_matches("brain_ep_")
                        .trim_end_matches(".msgpack");

                    if let Ok(ep) = num_part.parse::<u32>() {
                        if ep >= max_ep {
                            max_ep = ep;
                            found = true;
                        }
                    }
                }
            }
        }
    }

    if found {
        Some(max_ep)
    } else {
        None
    }
}

// Add a dummy step function wrapper if needed, but ThreesEnv::step is private?
// Checking threes_env.rs...
// `step` is private `fn step`.
// `train_step` is public.
// We need a public way to advance the game without training.
// `train.rs` uses `step` inside `train_step`.
// In `run_evaluation_training`, it uses `local_env.game.move_dir(action_dir)`.
// `move_dir` returns `(bool, Vec<u8>)`.
// So we should do what `run_evaluation_training` does: manually call `move_dir`.
/*
                    let action_dir = match action {
                        0 => Direction::Up,
                        1 => Direction::Down,
                        2 => Direction::Left,
                        3 => Direction::Right,
                        _ => unreachable!(),
                    };

                    let (moved, _) = local_env.game.move_dir(action_dir);
                    if !moved {
                        break;
                    }
*/
// Okay, I will use this pattern in the code above.
