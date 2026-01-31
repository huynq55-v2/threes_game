use crate::game::{Direction, Game};
use crossterm::{
    cursor,
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
        MouseEvent, MouseEventKind,
    },
    style::{self, Color, Stylize},
    terminal::{self, ClearType},
    ExecutableCommand, QueueableCommand,
};

pub enum InputEvent {
    Dir(Direction),
    Undo,
    Redo,
    Quit,
}
use std::io::{self, Write};

pub struct ConsoleUI;

impl ConsoleUI {
    pub fn init() -> io::Result<()> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(terminal::EnterAlternateScreen)?;
        stdout.execute(cursor::Hide)?;
        stdout.execute(EnableMouseCapture)?;
        Ok(())
    }

    pub fn cleanup() -> io::Result<()> {
        let mut stdout = io::stdout();
        stdout.execute(DisableMouseCapture)?;
        stdout.execute(cursor::Show)?;
        stdout.execute(terminal::LeaveAlternateScreen)?;
        terminal::disable_raw_mode()?;
        Ok(())
    }

    pub fn print_game_state(
        game: &Game,
        _valid_moves: &[Direction],
        message: Option<&str>,
    ) -> io::Result<()> {
        let mut stdout = io::stdout();
        stdout.queue(terminal::Clear(ClearType::All))?;
        stdout.queue(cursor::MoveTo(0, 0))?;

        // 1. Header & Score
        let header = format!(" THREES | Score: {} | Moves: {}", game.score, game.num_move);
        stdout.queue(style::Print(format!("\r\n {}\r\n", header.bold())))?;

        // 2. Next Hint
        stdout.queue(style::Print(" Next: "))?;
        if game.hints.is_empty() {
            stdout.queue(style::Print("(?)"))?;
        } else {
            for hint in &game.hints {
                let val_str = format!(" {} ", hint);
                let styled = Self::style_tile_content(&val_str, *hint);
                stdout.queue(style::Print(styled))?;
                stdout.queue(style::Print(" "))?;
            }
        }
        stdout.queue(style::Print("\r\n\r\n"))?;

        // 3. Draw Board (Big Squares)
        // Constants for layout
        let cell_width = 9; // 9 chars wide (internal 7 spaces is good, but let's do 9 total chars including 1 border?)
                            // Let's do Box Drawing.
                            // ┌───────┐
                            // │       │
                            // │ 12345 │  <- 7 chars internal
                            // │       │
                            // └───────┘
        let internal_w = 7;

        // Helper to draw a horizontal line
        // top:    ┌───────┬─────── ... ┐
        // mid:    ├───────┼─────── ... ┤
        // bottom: └───────┴─────── ... ┘
        let draw_h_line = |out: &mut io::Stdout,
                           left: char,
                           mid: char,
                           cross: char,
                           right: char|
         -> io::Result<()> {
            out.queue(style::Print(" "))?; // Margin
            out.queue(style::Print(left))?;
            for c in 0..4 {
                for _ in 0..internal_w {
                    out.queue(style::Print(mid))?;
                }
                if c < 3 {
                    out.queue(style::Print(cross))?;
                }
            }
            out.queue(style::Print(right))?;
            out.queue(style::Print("\r\n"))?;
            Ok(())
        };

        // Draw Top Border
        draw_h_line(&mut stdout, '┌', '─', '┬', '┐')?;

        for r in 0..4 {
            // We need 3 lines for the cell body
            // Line 1: Padding
            // Line 2: Number
            // Line 3: Padding

            for line_idx in 0..3 {
                stdout.queue(style::Print(" │"))?; // Leftmost border
                for c in 0..4 {
                    let val = game.board[r][c].value;

                    // Prepare content string
                    let content_str = if line_idx == 1 {
                        // Value line
                        if val == 0 {
                            " ".repeat(internal_w)
                        } else {
                            format!("{:^width$}", val, width = internal_w)
                        }
                    } else {
                        // Padding line
                        " ".repeat(internal_w)
                    };

                    // Style the content background
                    let styled = Self::style_tile_content(&content_str, val);
                    stdout.queue(style::Print(styled))?;

                    // Right separator (internal or end)
                    // If it's the last col, we print the final border later
                    if c < 3 {
                        stdout.queue(style::Print("│"))?;
                    }
                }
                stdout.queue(style::Print("│\r\n"))?; // Rightmost border
            }

            // Draw Mid or Bottom Border
            if r < 3 {
                draw_h_line(&mut stdout, '├', '─', '┼', '┤')?;
            } else {
                draw_h_line(&mut stdout, '└', '─', '┴', '┘')?;
            }
        }

        // 4. Instructions / Message
        stdout.queue(style::Print("\r\n"))?;
        if let Some(msg) = message {
            // Red message if error-like?
            stdout.queue(style::Print(format!(" {}\r\n\r\n", msg.red().bold())))?;
        } else {
            stdout.queue(style::Print("\r\n\r\n"))?;
        }

        stdout.queue(style::Print(" Controls: ".grey()))?;
        stdout.queue(style::Print("[Arrows/WASD] Move  [Q] Quit\r\n"))?;

        stdout.flush()?;
        Ok(())
    }

    fn style_tile_content(content: &str, val: u32) -> style::StyledContent<&str> {
        let text = content;
        match val {
            0 => text.reset(), // Empty
            1 => text.with(Color::White).on(Color::Blue),
            2 => text.with(Color::White).on(Color::Red),
            // 3..=max usually black on white
            _ => text.with(Color::Black).on(Color::White),
        }
    }

    pub fn get_input() -> io::Result<InputEvent> {
        loop {
            if event::poll(std::time::Duration::from_millis(50))? {
                let event = event::read()?;
                match event {
                    Event::Key(KeyEvent {
                        code, modifiers, ..
                    }) => {
                        if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
                            return Ok(InputEvent::Quit);
                        }
                        match code {
                            KeyCode::Up | KeyCode::Char('w') | KeyCode::Char('W') => {
                                return Ok(InputEvent::Dir(Direction::Up))
                            }
                            KeyCode::Down | KeyCode::Char('s') | KeyCode::Char('S') => {
                                return Ok(InputEvent::Dir(Direction::Down))
                            }
                            KeyCode::Left | KeyCode::Char('a') | KeyCode::Char('A') => {
                                return Ok(InputEvent::Dir(Direction::Left))
                            }
                            KeyCode::Right | KeyCode::Char('d') | KeyCode::Char('D') => {
                                return Ok(InputEvent::Dir(Direction::Right))
                            }
                            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                                return Ok(InputEvent::Quit)
                            }
                            _ => {}
                        }
                    }
                    Event::Mouse(MouseEvent { kind, .. }) => match kind {
                        MouseEventKind::ScrollUp => return Ok(InputEvent::Undo),
                        MouseEventKind::ScrollDown => return Ok(InputEvent::Redo),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    pub fn wait_for_enter() -> io::Result<()> {
        loop {
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                    if code == KeyCode::Enter || code == KeyCode::Char(' ') {
                        return Ok(());
                    }
                    if code == KeyCode::Char('q') {
                        // Ideally we check this in the loop, but for now simple Wait
                        return Ok(());
                    }
                }
            }
        }
    }
}
