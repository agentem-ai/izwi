//! Streaming audio buffer for Voxtral Realtime.
//!
//! Manages audio buffering with look-ahead and look-back for streaming transcription.

/// Buffer for realtime streaming audio
pub struct VoxtralRealtimeBuffer {
    sampling_rate: usize,
    look_ahead: usize,
    look_back: usize,
    streaming_size: usize,
    start: usize,
    end: usize,
    buffer: Vec<f32>,
    filled_len: usize,
    pre_allocate_size: usize,
}

impl VoxtralRealtimeBuffer {
    /// Create new buffer with audio config
    pub fn new(
        sampling_rate: usize,
        streaming_look_ahead_ms: f32,
        streaming_look_back_ms: f32,
        transcription_delay_ms: f32,
        frame_rate: f32,
    ) -> Self {
        let look_ahead = ((sampling_rate as f32 * streaming_look_ahead_ms) / 1000.0) as usize;
        let look_back = ((sampling_rate as f32 * streaming_look_back_ms) / 1000.0) as usize;
        let streaming_size = ((sampling_rate as f32 * 1000.0) / (frame_rate * 1000.0)) as usize;
        let streaming_delay = ((sampling_rate as f32 * transcription_delay_ms) / 1000.0) as usize;
        
        let pre_allocate_size = 30 * sampling_rate; // 30 seconds
        let buffer = vec![0.0f32; pre_allocate_size];
        
        Self {
            sampling_rate,
            look_ahead,
            look_back,
            streaming_size,
            start: 0,
            end: streaming_delay + streaming_size,
            buffer,
            filled_len: 0,
            pre_allocate_size,
        }
    }
    
    fn get_len_in_samples(&self, len_in_ms: f32) -> usize {
        ((self.sampling_rate as f32 * len_in_ms) / 1000.0) as usize
    }
    
    /// Start index including look-back
    pub fn start_idx(&self) -> usize {
        self.start.saturating_sub(self.look_back)
    }
    
    /// End index including look-ahead
    pub fn end_idx(&self) -> usize {
        self.end + self.look_ahead
    }
    
    /// Check if enough audio is available for processing
    pub fn is_audio_complete(&self) -> bool {
        self.filled_len >= self.end_idx()
    }
    
    /// Write audio chunk to buffer
    pub fn write_audio(&mut self, audio: &[f32]) {
        let put_end = self.filled_len + audio.len();
        
        if put_end > self.pre_allocate_size {
            self.allocate_new_buffer();
        }
        
        self.buffer[self.filled_len..self.filled_len + audio.len()].copy_from_slice(audio);
        self.filled_len += audio.len();
    }
    
    fn allocate_new_buffer(&mut self) {
        let mut new_buffer = vec![0.0f32; self.pre_allocate_size];
        let left_to_copy = self.filled_len.saturating_sub(self.start_idx());
        
        if left_to_copy > 0 {
            new_buffer[..left_to_copy].copy_from_slice(
                &self.buffer[self.start_idx()..self.filled_len]
            );
        }
        
        self.buffer = new_buffer;
        self.filled_len = left_to_copy;
        self.start = self.look_back;
        self.end = self.start + self.streaming_size;
    }
    
    /// Read audio chunk for processing (with look-ahead/look-back)
    pub fn read_audio(&mut self) -> Option<Vec<f32>> {
        if !self.is_audio_complete() {
            return None;
        }
        
        let audio = self.buffer[self.start_idx()..self.end_idx()].to_vec();
        self.start = self.end;
        self.end += self.streaming_size;
        
        Some(audio)
    }
}
