/** Keep the Starlight page title as the only H1, including historical content. */
export default function remarkSingleTitle() {
  return (tree) => {
    const hasBodyTitle = tree.children.some((node) => node.type === 'heading' && node.depth === 1);
    if (!hasBodyTitle) return;
    for (const node of tree.children) {
      if (node.type === 'heading') node.depth = Math.min(6, node.depth + 1);
    }
  };
}
