import { html, fixture, expect } from '@open-wc/testing';

import '../view-tos.js';

describe('ViewTos', () => {
  it('has a default title "Hey there" and counter 5', async () => {
    const el = await fixture(html`
      <view-tos></view-tos>
    `);

    expect(el.title).to.equal('Hey there');
    expect(el.counter).to.equal(5);
  });

  it('increases the counter on button click', async () => {
    const el = await fixture(html`
      <view-tos></view-tos>
    `);
    el.shadowRoot.querySelector('button').click();

    expect(el.counter).to.equal(6);
  });

  it('can override the title via attribute', async () => {
    const el = await fixture(html`
      <view-tos title="attribute title"></view-tos>
    `);

    expect(el.title).to.equal('attribute title');
  });

  it('shows initially the text "hey there Nr. 5!" and an "increment" button', async () => {
    const el = await fixture(html`
      <view-tos></view-tos>
    `);

    expect(el).shadowDom.to.equalSnapshot();
  });

  it('passes the a11y audit', async () => {
    const el = await fixture(html`
      <view-tos></view-tos>
    `);

    await expect(el).shadowDom.to.be.accessible();
  });
});
