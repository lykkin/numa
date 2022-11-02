use std::collections::HashMap;

#[derive(Clone)]
pub struct Tracer {
    pub calls: HashMap<String, usize>
}

impl Tracer {
    pub fn new() -> Self {
        Self {
            calls: HashMap::new()
        }
    }
    pub fn increment_call(self: &mut Self, name: String, increment: usize) {
        self.calls
            .entry(name)
            .and_modify(|c| *c += increment)
            .or_insert(0);
    }

    pub fn print(self: &Self) {
        // borrow the tracer back
        for (k, v) in self.calls.iter() {
            println!("{}: {}", k, v);
        }
    }
}