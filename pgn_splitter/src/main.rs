use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// Define Elo thresholds for the 6 compartments
fn get_compartment_index(avg_elo: u32) -> usize {
    if avg_elo < 1200 { 0 }
    else if avg_elo < 1500 { 1 }
    else if avg_elo < 1800 { 2 }
    else if avg_elo < 2100 { 3 }
    else if avg_elo < 2400 { 4 }
    else { 5 }
}

fn main() -> io::Result<()> {
    let input_path = "lichess_db.pgn"; 
    let file = File::open(input_path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer size for fast I/O

    let file_names = [
        "comp_1200_under.pgn",
        "comp_1200_1500.pgn",
        "comp_1500_1800.pgn",
        "comp_1800_2100.pgn",
        "comp_2100_2400.pgn",
        "comp_2400_above.pgn",
    ];

    let mut writers: Vec<BufWriter<File>> = file_names
        .iter()
        .map(|name| {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .append(true) // <-- This keeps existing data and adds to the end
                .open(name)
                .unwrap();
            BufWriter::with_capacity(1024 * 1024, file)
        })
        .collect();

    // Reusable buffers to minimize heap allocations during loops
    let mut game_buffer = String::with_capacity(4096);
    
    // Track Elo states explicitly: Some(value), None (unparsed yet), or Unknown (the "?" case)
    #[derive(Clone, Copy, PartialEq)]
    enum EloState {
        Unparsed,
        Known(u32),
        Unknown,
    }

    let mut white_elo = EloState::Unparsed;
    let mut black_elo = EloState::Unparsed;
    let mut game_counter = 0;
    let mut skipped_counter = 0;

    println!("Starting optimization split with Elo safeguards...");

    for line_result in reader.lines() {
        let line = line_result?;
        
        game_buffer.push_str(&line);
        game_buffer.push('\n');

        // Fast prefix matching for White Elo
        if line.starts_with("[WhiteElo \"") {
            if let Some(end) = line.get(11..) {
                if let Some(idx) = end.find('"') {
                    let val = &end[..idx];
                    if val == "?" {
                        white_elo = EloState::Unknown;
                    } else if let Ok(num) = val.parse::<u32>() {
                        white_elo = EloState::Known(num);
                    } else {
                        white_elo = EloState::Unknown;
                    }
                }
            }
        } 
        // Fast prefix matching for Black Elo
        else if line.starts_with("[BlackElo \"") {
            if let Some(end) = line.get(11..) {
                if let Some(idx) = end.find('"') {
                    let val = &end[..idx];
                    if val == "?" {
                        black_elo = EloState::Unknown;
                    } else if let Ok(num) = val.parse::<u32>() {
                        black_elo = EloState::Known(num);
                    } else {
                        black_elo = EloState::Unknown;
                    }
                }
            }
        }
        // Check if the game block has ended (empty line following the moves)
        // Ensure both Elos have processed out of the Unparsed state
        else if line.trim().is_empty() && white_elo != EloState::Unparsed && black_elo != EloState::Unparsed {
            
            // Resolve Elo based on your safeguard instructions
            let resolved_avg_elo: Option<u32> = match (white_elo, black_elo) {
                (EloState::Known(w), EloState::Known(b)) => Some((w + b) / 2),
                (EloState::Known(w), EloState::Unknown) => Some(w),
                (EloState::Unknown, EloState::Known(b)) => Some(b),
                (EloState::Unknown, EloState::Unknown) => None, // Both are anonymous -> Skip
                _ => None, // Safety fallback
            };

            if let Some(avg_elo) = resolved_avg_elo {
                let comp_idx = get_compartment_index(avg_elo);
                writers[comp_idx].write_all(game_buffer.as_bytes())?;
                game_counter += 1;
            } else {
                skipped_counter += 1;
            }
            
            // Fast reset for next game loop
            game_buffer.clear();
            white_elo = EloState::Unparsed;
            black_elo = EloState::Unparsed;
            
            if (game_counter + skipped_counter) % 500_000 == 0 {
                println!("Processed {} games (Skipped {} anonymous games)...", game_counter, skipped_counter);
            }
        }
    }

    // Flush RAM buffers into storage
    for mut writer in writers {
        writer.flush()?;
    }

    println!("Done!");
    println!("Total games written: {}", game_counter);
    println!("Total games skipped: {}", skipped_counter);
    Ok(())
}