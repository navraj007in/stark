//! Comment trivia cursor used while printing.
//!
//! Comments are collected separately from the token stream
//! (`lexer::tokenize_with_comments`) and re-attached to the nearest AST node
//! by source position as the printer walks the tree. This cursor consumes
//! them in order (they are already position-sorted, since the lexer emits
//! them as it scans left to right).

use crate::lexer::Comment;

pub struct CommentStream<'a> {
    comments: &'a [Comment],
    pos: usize,
}

impl<'a> CommentStream<'a> {
    pub fn new(comments: &'a [Comment]) -> Self {
        CommentStream { comments, pos: 0 }
    }

    /// Consume and return the next comment if it starts strictly before
    /// `before_pos` (i.e. it lies ahead of the AST node about to be
    /// printed).
    pub fn take_before(&mut self, before_pos: u32) -> Option<Comment> {
        let next = self.comments.get(self.pos)?;
        if next.span.lo < before_pos {
            self.pos += 1;
            Some(*next)
        } else {
            None
        }
    }

    pub fn peek(&self) -> Option<Comment> {
        self.comments.get(self.pos).copied()
    }

    /// Consume the comment [`peek`](Self::peek) just returned.
    pub fn advance(&mut self) {
        if self.pos < self.comments.len() {
            self.pos += 1;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.comments.len()
    }

    /// Drain and return every remaining comment; used as a no-loss safety
    /// net at the end of a block/program so a comment in a position the
    /// printer doesn't specifically attach (e.g. inside an expression) is
    /// still emitted rather than silently dropped.
    pub fn take_rest(&mut self) -> &'a [Comment] {
        let rest = &self.comments[self.pos.min(self.comments.len())..];
        self.pos = self.comments.len();
        rest
    }
}
