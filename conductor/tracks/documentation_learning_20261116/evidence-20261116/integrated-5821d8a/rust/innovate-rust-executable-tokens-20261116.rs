use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use std::{env, fs, str::FromStr};

fn without_docs(input: TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let mut output = TokenStream::new();
    let mut index = 0;
    while index < tokens.len() {
        if matches!(&tokens[index], TokenTree::Punct(p) if p.as_char() == '#') {
            let mut group_index = index + 1;
            if matches!(tokens.get(group_index), Some(TokenTree::Punct(p)) if p.as_char() == '!') {
                group_index += 1;
            }
            if let Some(TokenTree::Group(group)) = tokens.get(group_index) {
                let attr: Vec<_> = group.stream().into_iter().collect();
                if group.delimiter() == Delimiter::Bracket
                    && attr.len() == 3
                    && matches!(&attr[0], TokenTree::Ident(name) if name == "doc")
                    && matches!(&attr[1], TokenTree::Punct(p) if p.as_char() == '=')
                    && matches!(&attr[2], TokenTree::Literal(_))
                {
                    index = group_index + 1;
                    continue;
                }
            }
        }
        let token = match &tokens[index] {
            TokenTree::Group(group) => TokenTree::Group(Group::new(group.delimiter(), without_docs(group.stream()))),
            token => token.clone(),
        };
        output.extend([token]);
        index += 1;
    }
    output
}

fn main() {
    let file = env::args().nth(1).expect("Rust source path");
    let source = fs::read_to_string(file).expect("source must be UTF-8");
    let parsed = TokenStream::from_str(&source).expect("Rust tokenization must succeed");
    println!("{}", without_docs(parsed));
}
