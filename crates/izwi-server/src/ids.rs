pub fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}
