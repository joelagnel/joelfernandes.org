# Joel Fernandes Website - Brendan Gregg Style

A redesign of [joelfernandes.org](https://www.joelfernandes.org) using the exact same design as [Brendan Gregg's website](https://www.brendangregg.com).

## Design

This site uses **Brendan Gregg's exact CSS** (`page.css`) and matching HTML structure:

- **Background:** Light pink (#ffe0e0)
- **Layout:** Three columns (nav | content | sidebar)
- **Fonts:** Garamond/Times serif body, Avant Garde/Helvetica headers
- **Colors:** Blue links (#00a), purple visited (#a0a)
- **Responsive:** Sidebars hide on narrow screens (<1050px)

## Quick Start

```bash
cd joelfernandes-brendangregg-style

# Serve locally
python3 -m http.server 4445 --bind 0.0.0.0

# Visit http://localhost:4445
```

## Structure

```
.
├── page.css              # Brendan Gregg's exact CSS
├── index.html            # Homepage
├── bio.html              # Biography
├── overview.html         # Start Here page
├── blog/
│   └── index.html        # Blog listing
├── resources/
│   ├── index.html        # Talks page
│   └── *.pdf             # Slide PDFs
├── joel/
│   ├── index.html        # Resume page
│   └── joel-resume.pdf   # Resume PDF
├── Images/
│   └── joel_photo.jpg    # Profile photo
├── linuxperf.html        # Linux Performance page
├── rcu.html              # RCU page
├── tracing.html          # Tracing page
├── schedulers.html       # Schedulers page
├── memory-ordering.html  # Memory Ordering page
├── gpu.html              # GPU Drivers page
├── books.html            # Books page
└── sites.html            # Other Sites page
```

## Comparison

| Feature | Brendan Gregg | Joel Fernandes |
|---------|---------------|----------------|
| Background | #ffe0e0 (pink) | #ffe0e0 (pink) |
| Layout | 3-column | 3-column |
| Left nav | Fixed, gray | Fixed, gray |
| Right sidebar | Books + posts | Photo + posts |
| Fonts | Garamond | Garamond |
| Mobile | Responsive | Responsive |

## Customization

Replace `Images/joel_photo.jpg` with your actual photo.

## Credits

- Design: [Brendan Gregg](https://www.brendangregg.com)
- Content: [Joel Fernandes](https://www.joelfernandes.org)
