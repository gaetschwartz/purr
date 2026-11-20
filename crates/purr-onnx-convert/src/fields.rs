//! Field-by-field reading of a protobuf stream.

use crate::error::ConvertError;
use protobuf::rt::WireType;
use protobuf::{CodedInputStream, Message};

pub struct Field {
    pub number: u32,
    pub wire_type: WireType,
}

pub fn read_failed(source: protobuf::Error) -> ConvertError {
    ConvertError::Read { source }
}

pub fn next_field(input: &mut CodedInputStream) -> Result<Option<Field>, ConvertError> {
    let Some(tag) = input.read_raw_tag_or_eof().map_err(read_failed)? else {
        return Ok(None);
    };
    let number = tag >> 3;
    let wire_type = WireType::new(tag & 0b111).ok_or(ConvertError::WireType {
        field: number,
        wire_type: tag & 0b111,
    })?;
    Ok(Some(Field { number, wire_type }))
}

/// A message whose fields are gathered as raw values while the stream is walked, so that the
/// fields the walker handles itself never pass through memory twice.
pub struct Remainder<M: Message> {
    message: M,
}

impl<M: Message> Remainder<M> {
    pub fn new() -> Self {
        Self { message: M::new() }
    }

    pub fn keep(
        &mut self,
        input: &mut CodedInputStream,
        field: &Field,
    ) -> Result<(), ConvertError> {
        let value = input.read_unknown(field.wire_type).map_err(read_failed)?;
        self.message
            .mut_unknown_fields()
            .add_value(field.number, value);
        Ok(())
    }

    pub fn keep_bytes(&mut self, number: u32, bytes: Vec<u8>) {
        self.message
            .mut_unknown_fields()
            .add_value(number, protobuf::UnknownValue::LengthDelimited(bytes));
    }

    /// Decodes a copy of the values gathered so far as `M`.
    pub fn peek(&self) -> Result<M, ConvertError> {
        let bytes = self
            .message
            .write_to_bytes()
            .map_err(|source| ConvertError::Encode { source })?;
        M::parse_from_bytes(&bytes).map_err(read_failed)
    }

    /// Decodes the gathered values as `M`.
    pub fn finish(self) -> Result<M, ConvertError> {
        let bytes = self
            .message
            .write_to_bytes()
            .map_err(|source| ConvertError::Encode { source })?;
        let decoded = M::parse_from_bytes(&bytes).map_err(read_failed)?;
        if decoded.unknown_fields().iter().next().is_some() {
            let numbers: Vec<String> = decoded
                .unknown_fields()
                .iter()
                .map(|(number, _)| number.to_string())
                .collect();
            return Err(ConvertError::UnknownField {
                message: format!("field numbers {}", numbers.join(", ")),
            });
        }
        Ok(decoded)
    }
}
